"""Acquire every input into `data/raw/<source>/` and write `data/manifest.csv`.

Run with `uv run python -m uhi_poverty.acquire` (or `make data`). Files already
on disk are reused, so the command is safe to repeat. The Census API key is read
from `CENSUS_API_KEY` (environment or the gitignored `.env`) and never written
anywhere: request URLs are redacted before they reach a log line or the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from . import geo, paths

log = logging.getLogger("uhi_poverty.acquire")

# S1: the SUHI database, Mendeley Data x9mv4krnm2 version 3. The SHA-256 is the one Mendeley
# publishes for the file.
MENDELEY_URL = "https://data.mendeley.com/datasets/x9mv4krnm2/3"
SUHI_URL = ("https://data.mendeley.com/public-files/datasets/x9mv4krnm2/files/"
            "54876bc0-ebda-48c6-bcdc-0bbf8e7b485f/file_downloaded")
SUHI_SHA256 = "7c0c39af227923dc1aaf5948bc43c79e6be7284d20c5c8e7a3107748e4a25585"
SUHI_BYTES = 45_925_109
# S3: HUD's Low Poverty Index (AFFHT0006), the CSV export of HUD Open Data item
# 3419cb4c7aa144b2bc54671f58b580f4, layer 20.
LPI_URL = ("https://hudgis-hud.opendata.arcgis.com/api/download/v1/items/"
           "3419cb4c7aa144b2bc54671f58b580f4/csv?layers=20")
LPI_PAGE = "https://hudgis-hud.opendata.arcgis.com/datasets/HUD::low-poverty-index"
LPI_SHA256 = "64f4dc2f297faa721056f465546e116268d5fd879daec4c8b551c328108421e6"
LPI_BYTES = 8_761_721

ACS_URL = "https://api.census.gov/data/2019/acs/acs5"
ACS_VARS = ["B17001_001E", "B17001_001M", "B17001_002E", "B17001_002M", "B01003_001E"]
GENZ = "https://www2.census.gov/geo/tiger/GENZ2019/shp"
GAZ19_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/"
             "2019_Gaz_tracts_national.zip")
SVI_URL = "https://svi.cdc.gov/Documents/Data/2018/csv/states/SVI_2018_US.csv"

PUBLIC_DOMAIN = "U.S. Government work (public domain)"

class AcquisitionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from `.env` into the environment (without overriding it)."""
    path = path or paths.ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def census_key() -> str | None:
    load_dotenv()
    return os.environ.get("CENSUS_API_KEY") or None


def redact(url: str) -> str:
    """Drop the `key` query parameter from a URL."""
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k.lower() != "key"]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query, safe=":*,")))


def fetch(url: str, dest: Path, retries: int = 4, timeout: int = 120) -> None:
    """Download `url` to `dest` with retries. Error messages carry the redacted URL only."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    safe = redact(url)
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "uhi-poverty/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            code = getattr(exc, "code", None)
            last = f"HTTP {code}" if code else type(exc).__name__
            log.warning("attempt %d/%d failed for %s (%s)", attempt, retries, safe, last)
            if code is not None and 400 <= code < 500 and code != 429:
                break
            time.sleep(2 * attempt)
    if tmp.exists():
        tmp.unlink()
    raise AcquisitionError(f"could not download {safe}: {last}") from None


def manifest_row(source_id: str, dataset: str, origin: str, path: Path, licence: str,
                 retrieved: str | None = None) -> dict:
    return {
        "id": source_id,
        "dataset": dataset,
        "origin": origin,
        "file": paths.rel(path),
        "retrieved": retrieved or date.today().isoformat(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "licence": licence,
    }


def previous_dates() -> dict[tuple[str, str], str]:
    """Retrieval dates already in the manifest, so re-runs keep the original date."""
    if not paths.MANIFEST.exists():
        return {}
    old = pd.read_csv(paths.MANIFEST, dtype=str)
    return {(r.file, r.sha256): r.retrieved for r in old.itertuples()}


# --------------------------------------------------------------------------- sources


def acquire_verified(source_id: str, url: str, dest: Path, sha: str, n_bytes: int) -> Path:
    """Download a source file once and check it against its pinned size and SHA-256."""
    acquire_url(url, dest)
    digest = sha256(dest)
    if digest != sha or dest.stat().st_size != n_bytes:
        raise AcquisitionError(f"{source_id}: {paths.rel(dest)} has SHA-256 {digest}, expected {sha}")
    log.info("%s matches its pinned SHA-256", source_id)
    return dest


def acquire_acs(states: list[str], key: str) -> list[Path]:
    """One ACS 2015–2019 call per state, cached as raw JSON."""
    out = []
    for fips in states:
        dest = paths.ACS_DIR / f"acs5_2019_B17001_{fips}.json"
        if not dest.exists():
            query = urllib.parse.urlencode(
                {"get": "NAME," + ",".join(ACS_VARS), "for": "tract:*", "in": f"state:{fips}",
                 "key": key},
                safe=":*,",
            )
            fetch(f"{ACS_URL}?{query}", dest)
            json.loads(dest.read_text())  # fail loudly on an HTML error page
            log.info("ACS state %s: %d bytes", fips, dest.stat().st_size)
        out.append(dest)
    return out


def acs_url_for(fips: str) -> str:
    query = urllib.parse.urlencode(
        {"get": "NAME," + ",".join(ACS_VARS), "for": "tract:*", "in": f"state:{fips}"},
        safe=":*,",
    )
    return f"{ACS_URL}?{query}"


def acquire_url(url: str, dest: Path) -> Path:
    if not dest.exists():
        fetch(url, dest)
        log.info("downloaded %s (%d bytes)", paths.rel(dest), dest.stat().st_size)
    return dest


# --------------------------------------------------------------------------- main


def run(poverty_source: str = "acs") -> pd.DataFrame:
    dates = previous_dates()
    rows: list[dict] = []

    def add(source_id, dataset, origin, path, licence):
        row = manifest_row(source_id, dataset, origin, path, licence)
        row["retrieved"] = dates.get((row["file"], row["sha256"]), row["retrieved"])
        rows.append(row)

    s1 = acquire_verified("S1", SUHI_URL, paths.SUHI_CSV, SUHI_SHA256, SUHI_BYTES)
    add("S1", "SUHI database (Chakraborty et al., 2020), Census_UHI_US_Urbanized_recalculated.csv",
        MENDELEY_URL, s1, "CC BY 4.0")

    jurisdictions = sorted(geo.STATES)
    if poverty_source == "acs":
        key = census_key()
        if not key:
            raise AcquisitionError(
                "CENSUS_API_KEY is not set. Add it to .env or the environment, or rerun "
                "with --poverty-source svi2018 (logged as a deviation)."
            )
        for fips, path in zip(jurisdictions, acquire_acs(jurisdictions, key), strict=True):
            add("S2", f"ACS 2015-2019 5-year B17001/B01003, state {fips}", acs_url_for(fips),
                path, PUBLIC_DOMAIN)
    else:
        log.warning("DEVIATION: poverty source is CDC SVI 2018 EP_POV, not ACS 2015-2019")
        path = acquire_url(SVI_URL, paths.SVI_CSV)
        add("S2-fallback", "CDC/ATSDR SVI 2018 (ACS 2014-2018 EP_POV)", SVI_URL, path,
            PUBLIC_DOMAIN)

    s3 = acquire_verified("S3", LPI_URL, paths.LPI_CSV, LPI_SHA256, LPI_BYTES)
    add("S3", "HUD AFFH Low Poverty Index, AFFHT0006", LPI_PAGE, s3, PUBLIC_DOMAIN)

    for fips in jurisdictions:
        url = f"{GENZ}/cb_2019_{fips}_tract_500k.zip"
        path = acquire_url(url, paths.TRACTS_DIR / f"cb_2019_{fips}_tract_500k.zip")
        add("S4", f"Cartographic boundary tracts 2019 1:500k, state {fips}", url, path,
            PUBLIC_DOMAIN)

    url = f"{GENZ}/cb_2019_us_state_500k.zip"
    add("S5", "Cartographic boundary states 2019 1:500k", url, acquire_url(url, paths.STATES_ZIP),
        PUBLIC_DOMAIN)
    add("S6", "Census 2019 Gazetteer, tracts (2010 tract codes)", GAZ19_URL,
        acquire_url(GAZ19_URL, paths.GAZ_2019), PUBLIC_DOMAIN)

    manifest = pd.DataFrame(rows)
    paths.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths.MANIFEST, index=False)
    log.info("manifest: %d files, %.1f MB", len(manifest), manifest["bytes"].sum() / 1e6)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poverty-source", choices=["acs", "svi2018"], default="acs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    run(args.poverty_source)


if __name__ == "__main__":
    main()
