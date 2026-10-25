# 6 Figures

The paper's figures. Each has its own script, `results/` files holding every number it shows,
and a README written from those results. The national heat map is produced by
[`1_Geographical_Scope/Heatmap/`](../1_Geographical_Scope/Heatmap/).

| Figure | What it shows | Image | README |
|---|---|---|---|
| Pipeline diagram | Datasets, the two classification tasks, models and validation | [`pipeline.png`](results/pipeline.png) | [README_pipeline](README_pipeline.md) |
| Algorithm comparison | Eight algorithms × four metrics under leave-one-city-out, for the UHI task (a) and the poverty task (b) | [`algorithms.png`](results/algorithms.png) | [README_algorithms](README_algorithms.md) |
| Confusion matrices by city, UHI | The UHI model in the ten largest held-out cities | [`confusion_by_city_uhi.png`](results/confusion_by_city_uhi.png) | [README_confusion_by_city](README_confusion_by_city.md) |
| Confusion matrices by city, poverty | The poverty model in the ten largest held-out cities | [`confusion_by_city_poverty.png`](results/confusion_by_city_poverty.png) | [README_confusion_by_city](README_confusion_by_city.md) |
| SHAP beeswarm | SHAP values of the top 10 features per task | [`shap_beeswarm.png`](results/shap_beeswarm.png) | [README_shap_beeswarm](README_shap_beeswarm.md) |
| Chi-squared heatmap | The 2×2 table of SUHI class against poverty class, with the odds ratio | [`chi_squared_heatmap.png`](results/chi_squared_heatmap.png) | [README_chi_squared_heatmap](README_chi_squared_heatmap.md) |

The figures read their numbers from the analyses and check them against them: the chi-squared
heatmap from section 2.2
([`2.2_Chi_Squared_Median`](../2_Classification_Thresholds/2.2_Chi_Squared_Median/)), the
algorithm comparison and the confusion matrices by city from section 4.4
([`4.4_Metric_Panel_with_Uncertainty`](../4_Model_Specifications/4.4_Metric_Panel_with_Uncertainty/)),
and the SHAP beeswarm from section 5.2
([`5.2_Attribution_and_Stability`](../5_SHAP_Interpretation/5.2_Attribution_and_Stability/)).
The algorithm comparison also fits six further algorithms (decision tree, random forest, SVM,
KNN, CatBoost, XGBoost), so it needs `xgboost` and `catboost`.

Run a script from the repository root with `PYTHONPATH=src`, e.g.
`uv run python 6_Figures/chi_squared_heatmap.py`; `make figures` runs all five.
