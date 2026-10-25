# Final tuned models

Each file is the tuned procedure of section 4.6 applied once to all 54,243 tracts: inner
UA-grouped 5-fold cross-validation over every city chose the preprocessing options, the
parameters and the thresholds, and the pipeline was refitted on all tracts. The model type was
chosen by mean inner ROC-AUC in the nested runs, which uses training cities only. These models
have no held-out score of their own; their expected performance in a city they have not seen is
the nested leave-one-city-out estimate below.

| File | Target | Model | Inputs | τ_acc / τ_F1 | Accuracy, leave-one-city-out [95 % CI] | F1 (τ_F1) | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| `uhi_high_suhi.joblib` | High_UHI | Soft-voting ensemble, tuned weights | 28 | 0.45 / 0.38 | 0.835 [0.808, 0.856] | 0.844 | 0.913 |
| `poverty_high_pov.joblib` | High_Pov | Soft-voting ensemble, tuned weights | 64 | 0.53 / 0.41 | 0.706 [0.691, 0.723] | 0.736 | 0.779 |

The `.json` beside each file is its model card: the input columns in order, every chosen
setting, the thresholds, the selection table and the nested performance with intervals.

## Use

```python
import joblib
import pandas as pd

model = joblib.load("uhi_high_suhi.joblib")  # needs the repository's package: `uv sync`
X = pd.read_parquet("tracts.parquet")       # one row per tract, with model.columns present
probability = model.predict_proba(X)        # P(class 1)
label = model.predict(X)                    # τ_acc; model.predict(X, threshold="f1") for τ_F1
```

The UHI model takes no temperature input. Its inputs are vegetation, elevation, morphology and
tract geometry from the SUHI database and the 2019 Gazetteer, on 2010 tract boundaries. The
poverty model takes every SUHI-database column plus the same geometry. Both are cross-sectional
classifiers of the median split of the 2015–2019 sample; they are not forecasts and are not
validated for planning decisions (see the paper's limitations).
