# macOS can flag the editable-install .pth file as hidden, and Python 3.13 then
# skips it; putting src/ on the path keeps every target working regardless.
export PYTHONPATH := $(CURDIR)/src
PY := uv run python

.PHONY: all data build engine scope thresholds predictors models interpret shap tune tuned figures panel test lint clean-cache

all: data build engine models predictors thresholds scope interpret shap tune tuned figures panel test lint

data:
	$(PY) -m uhi_poverty.acquire

build:
	$(PY) -m uhi_poverty.build

engine:
	$(PY) -m uhi_poverty.validation

scope:
	$(PY) 1_Geographical_Scope/1.1_National_Focus/scope_summary.py
	$(PY) 1_Geographical_Scope/Heatmap/make_heatmap.py
	$(PY) 1_Geographical_Scope/1.3_Tract_Accounting/tract_accounting.py
	$(PY) 1_Geographical_Scope/1.4_Coverage_States_and_Urban_Areas/coverage.py
	$(PY) 1_Geographical_Scope/1.5_Large_Metro_Influence/metro_influence.py

thresholds:
	$(PY) 2_Classification_Thresholds/2.1_Median_Precedent/medians.py
	$(PY) 2_Classification_Thresholds/2.2_Chi_Squared_Median/chi_squared.py

predictors:
	$(PY) 3_UHI_Model_Predictors/3.1_UHI_Formula/verify_formula.py
	$(PY) 3_UHI_Model_Predictors/3.2_GBM_Without_LST/gbm_no_lst.py

models:
	$(PY) 4_Model_Specifications/4.1_Spatially_Grouped_Validation/grouped_validation.py
	$(PY) 4_Model_Specifications/4.2_Holdout_States_and_Major_Urban_Areas/holdout_vs_random.py
	$(PY) 4_Model_Specifications/4.3_Confusion_Matrices_and_Denominators/confusion_matrices.py
	$(PY) 4_Model_Specifications/4.4_Metric_Panel_with_Uncertainty/metric_panel.py
	$(PY) 4_Model_Specifications/4.5_Baselines/baselines.py

interpret:
	$(PY) -m uhi_poverty.interpret

shap:
	$(PY) 5_SHAP_Interpretation/5.1_Explained_Model/explained_model.py
	$(PY) 5_SHAP_Interpretation/5.2_Attribution_and_Stability/attribution.py

tune:
	$(PY) -m uhi_poverty.tuning

tuned:
	$(PY) 4_Model_Specifications/4.6_Tuned_Models_and_Ensemble/tuned_models.py

figures:
	$(PY) 6_Figures/pipeline.py
	$(PY) 6_Figures/algorithms.py
	$(PY) 6_Figures/confusion_by_city.py
	$(PY) 6_Figures/shap_beeswarm.py
	$(PY) 6_Figures/chi_squared_heatmap.py

# every model in one metric panel; reads the caches of models, predictors, tuned and figures
panel:
	$(PY) 4_Model_Specifications/4.7_All_Models_Metric_Panel/all_models.py

test:
	uv run pytest

lint:
	uv run ruff check

clean-cache:
	rm -rf data/interim/oof data/interim/interpret data/interim/tuned
