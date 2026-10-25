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

MENDELEY_SHA256 = "7c0c39af227923dc1aaf5948bc43c79e6be7284d20c5c8e7a3107748e4a25585"
MENDELEY_BYTES = 45_925_109
MENDELEY_URL = "https://data.mendeley.com/datasets/x9mv4krnm2/3"

ACS_URL = "https://api.census.gov/data/2019/acs/acs5"
ACS_VARS = ["B17001_001E", "B17001_001M", "B17001_002E", "B17001_002M", "B01003_001E"]
GENZ = "https://www2.census.gov/geo/tiger/GENZ2019/shp"
GAZ19_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/"
             "2019_Gaz_tracts_national.zip")
SVI_URL = "https://svi.cdc.gov/Documents/Data/2018/csv/states/SVI_2018_US.csv"

PUBLIC_DOMAIN = "U.S. Government work (public domain)"

# Source CSVs as they sit on the author's machine.
LOCAL_SOURCES = {
    "S1": ("US Urban Heat Island Data.csv", paths.SUHI_CSV),
    "S3": ("Low Poverty Index.csv", paths.LPI_CSV),
}


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


def acquire_local(source_id: str, source_dir: Path) -> Path:
    name, dest = LOCAL_SOURCES[source_id]
    src = source_dir / name
    if not dest.exists():
        if not src.exists():
            raise AcquisitionError(f"{source_id}: {paths.rel(src)} not found")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        log.info("%s copied from %s", source_id, paths.rel(src))
    return dest


def verify_suhi(path: Path) -> None:
    digest = sha256(path)
    if digest != MENDELEY_SHA256 or path.stat().st_size != MENDELEY_BYTES:
        raise AcquisitionError(
            f"S1 hash mismatch: {digest} (expected Mendeley {MENDELEY_SHA256})"
        )
    log.info("S1 hash matches Mendeley x9mv4krnm2 v3")


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


def run(poverty_source: str = "acs", source_dir: Path | None = None) -> pd.DataFrame:
    source_dir = source_dir or paths.SOURCE_DIR
    dates = previous_dates()
    rows: list[dict] = []

    def add(source_id, dataset, origin, path, licence):
        row = manifest_row(source_id, dataset, origin, path, licence)
        row["retrieved"] = dates.get((row["file"], row["sha256"]), row["retrieved"])
        rows.append(row)

    s1 = acquire_local("S1", source_dir)
    verify_suhi(s1)
    add("S1", "SUHI database (Chakraborty et al., 2020), Census_UHI_US_Urbanized_recalculated.csv",
        f"{MENDELEY_URL} (local copy {paths.rel(source_dir / LOCAL_SOURCES['S1'][0])})", s1,
        "CC BY 4.0")

    jurisdictions = sorted(geo.STATES) + [geo.PUERTO_RICO]
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

    s3 = acquire_local("S3", source_dir)
    add("S3", "HUD AFFH Low Poverty Index, AFFHT0006",
        f"HUD Open Data (local copy {paths.rel(source_dir / LOCAL_SOURCES['S3'][0])})", s3,
        PUBLIC_DOMAIN)

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
    parser.add_argument("--source-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    run(args.poverty_source, args.source_dir)


if __name__ == "__main__":
    main()
