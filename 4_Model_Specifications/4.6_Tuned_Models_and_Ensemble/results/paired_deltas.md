# Paired differences, Δ = a − b (95 % UA cluster-bootstrap CI)

| task | scheme | comparison | metric | a | b | delta | ci_low | ci_high | share_reps_above_0 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| A | loco | Logistic regression, tuned − LightGBM, fixed | accuracy | 0.838 | 0.823 | 0.015 | -0.010 | 0.056 | 0.652 |
| A | loco | Logistic regression, tuned − LightGBM, fixed | roc_auc | 0.902 | 0.910 | -0.008 | -0.020 | 0.006 | 0.134 |
| A | loco | Logistic regression, tuned (τ_F1) − LightGBM, fixed | f1 | 0.843 | 0.819 | 0.023 | -0.010 | 0.076 | 0.677 |
| A | loco | LightGBM, tuned − LightGBM, fixed | accuracy | 0.834 | 0.823 | 0.011 | -0.002 | 0.032 | 0.816 |
| A | loco | LightGBM, tuned − LightGBM, fixed | roc_auc | 0.910 | 0.910 | 0.001 | -0.005 | 0.006 | 0.547 |
| A | loco | LightGBM, tuned (τ_F1) − LightGBM, fixed | f1 | 0.840 | 0.819 | 0.021 | -0.001 | 0.057 | 0.959 |
| A | loco | XGBoost, tuned − LightGBM, fixed | accuracy | 0.834 | 0.823 | 0.010 | -0.002 | 0.029 | 0.886 |
| A | loco | XGBoost, tuned − LightGBM, fixed | roc_auc | 0.911 | 0.910 | 0.001 | -0.004 | 0.008 | 0.646 |
| A | loco | XGBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.846 | 0.819 | 0.026 | 0.002 | 0.068 | 0.997 |
| A | loco | CatBoost, tuned − LightGBM, fixed | accuracy | 0.833 | 0.823 | 0.010 | -0.002 | 0.028 | 0.862 |
| A | loco | CatBoost, tuned − LightGBM, fixed | roc_auc | 0.911 | 0.910 | 0.002 | -0.003 | 0.007 | 0.661 |
| A | loco | CatBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.840 | 0.819 | 0.021 | 0.002 | 0.051 | 0.995 |
| A | loco | Soft-voting ensemble, equal weights − LightGBM, fixed | accuracy | 0.833 | 0.823 | 0.010 | 0.000 | 0.023 | 0.974 |
| A | loco | Soft-voting ensemble, equal weights − LightGBM, fixed | roc_auc | 0.913 | 0.910 | 0.004 | -0.002 | 0.009 | 0.864 |
| A | loco | Soft-voting ensemble, equal weights (τ_F1) − LightGBM, fixed | f1 | 0.844 | 0.819 | 0.025 | 0.004 | 0.060 | 1.000 |
| A | loco | Soft-voting ensemble, tuned weights − LightGBM, fixed | accuracy | 0.835 | 0.823 | 0.012 | -0.001 | 0.031 | 0.950 |
| A | loco | Soft-voting ensemble, tuned weights − LightGBM, fixed | roc_auc | 0.913 | 0.910 | 0.003 | -0.002 | 0.009 | 0.800 |
| A | loco | Soft-voting ensemble, tuned weights (τ_F1) − LightGBM, fixed | f1 | 0.844 | 0.819 | 0.024 | 0.001 | 0.063 | 0.995 |
| A | loco | Soft-voting ensemble, equal weights − LightGBM, tuned | accuracy | 0.833 | 0.834 | -0.001 | -0.009 | 0.004 | 0.389 |
| A | loco | Soft-voting ensemble, equal weights − LightGBM, tuned | roc_auc | 0.913 | 0.910 | 0.003 | 0.001 | 0.005 | 1.000 |
| A | loco | Soft-voting ensemble, equal weights − XGBoost, tuned | accuracy | 0.833 | 0.834 | -0.001 | -0.006 | 0.002 | 0.364 |
| A | loco | Soft-voting ensemble, equal weights − XGBoost, tuned | roc_auc | 0.913 | 0.911 | 0.002 | 0.001 | 0.003 | 1.000 |
| A | loco | Soft-voting ensemble, equal weights − CatBoost, tuned | accuracy | 0.833 | 0.833 | -0.000 | -0.005 | 0.003 | 0.462 |
| A | loco | Soft-voting ensemble, equal weights − CatBoost, tuned | roc_auc | 0.913 | 0.911 | 0.002 | 0.000 | 0.004 | 0.984 |
| A | loco | Soft-voting ensemble, tuned weights − LightGBM, tuned | accuracy | 0.835 | 0.834 | 0.001 | -0.002 | 0.004 | 0.718 |
| A | loco | Soft-voting ensemble, tuned weights − LightGBM, tuned | roc_auc | 0.913 | 0.910 | 0.002 | 0.001 | 0.004 | 0.991 |
| A | loco | Soft-voting ensemble, tuned weights − XGBoost, tuned | accuracy | 0.835 | 0.834 | 0.001 | -0.000 | 0.003 | 0.922 |
| A | loco | Soft-voting ensemble, tuned weights − XGBoost, tuned | roc_auc | 0.913 | 0.911 | 0.002 | 0.000 | 0.003 | 0.987 |
| A | loco | Soft-voting ensemble, tuned weights − CatBoost, tuned | accuracy | 0.835 | 0.833 | 0.002 | -0.001 | 0.004 | 0.910 |
| A | loco | Soft-voting ensemble, tuned weights − CatBoost, tuned | roc_auc | 0.913 | 0.911 | 0.001 | -0.001 | 0.004 | 0.878 |
| A | ua5 | Logistic regression, tuned − LightGBM, fixed | accuracy | 0.835 | 0.823 | 0.012 | -0.013 | 0.052 | 0.652 |
| A | ua5 | Logistic regression, tuned − LightGBM, fixed | roc_auc | 0.900 | 0.908 | -0.008 | -0.019 | 0.005 | 0.109 |
| A | ua5 | Logistic regression, tuned (τ_F1) − LightGBM, fixed | f1 | 0.839 | 0.819 | 0.019 | -0.012 | 0.072 | 0.653 |
| A | ua5 | LightGBM, tuned − LightGBM, fixed | accuracy | 0.834 | 0.823 | 0.012 | -0.005 | 0.039 | 0.738 |
| A | ua5 | LightGBM, tuned − LightGBM, fixed | roc_auc | 0.910 | 0.908 | 0.002 | -0.005 | 0.010 | 0.672 |
| A | ua5 | LightGBM, tuned (τ_F1) − LightGBM, fixed | f1 | 0.840 | 0.819 | 0.021 | -0.001 | 0.060 | 0.927 |
| A | ua5 | XGBoost, tuned − LightGBM, fixed | accuracy | 0.828 | 0.823 | 0.005 | -0.003 | 0.017 | 0.738 |
| A | ua5 | XGBoost, tuned − LightGBM, fixed | roc_auc | 0.909 | 0.908 | 0.001 | -0.005 | 0.008 | 0.569 |
| A | ua5 | XGBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.843 | 0.819 | 0.024 | 0.000 | 0.064 | 0.983 |
| A | ua5 | CatBoost, tuned − LightGBM, fixed | accuracy | 0.840 | 0.823 | 0.018 | -0.001 | 0.050 | 0.889 |
| A | ua5 | CatBoost, tuned − LightGBM, fixed | roc_auc | 0.913 | 0.908 | 0.005 | -0.002 | 0.015 | 0.788 |
| A | ua5 | CatBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.846 | 0.819 | 0.027 | 0.002 | 0.071 | 0.996 |
| A | ua5 | Soft-voting ensemble, equal weights − LightGBM, fixed | accuracy | 0.838 | 0.823 | 0.015 | 0.000 | 0.041 | 0.979 |
| A | ua5 | Soft-voting ensemble, equal weights − LightGBM, fixed | roc_auc | 0.914 | 0.908 | 0.006 | -0.001 | 0.014 | 0.934 |
| A | ua5 | Soft-voting ensemble, equal weights (τ_F1) − LightGBM, fixed | f1 | 0.847 | 0.819 | 0.028 | 0.004 | 0.070 | 0.999 |
| A | ua5 | Soft-voting ensemble, tuned weights − LightGBM, fixed | accuracy | 0.837 | 0.823 | 0.015 | -0.000 | 0.040 | 0.965 |
| A | ua5 | Soft-voting ensemble, tuned weights − LightGBM, fixed | roc_auc | 0.912 | 0.908 | 0.005 | -0.002 | 0.013 | 0.853 |
| A | ua5 | Soft-voting ensemble, tuned weights (τ_F1) − LightGBM, fixed | f1 | 0.847 | 0.819 | 0.027 | 0.003 | 0.070 | 0.997 |
| A | ua5 | Soft-voting ensemble, equal weights − LightGBM, tuned | accuracy | 0.838 | 0.834 | 0.004 | -0.000 | 0.009 | 0.958 |
| A | ua5 | Soft-voting ensemble, equal weights − LightGBM, tuned | roc_auc | 0.914 | 0.910 | 0.004 | 0.001 | 0.006 | 0.996 |
| A | ua5 | Soft-voting ensemble, equal weights − XGBoost, tuned | accuracy | 0.838 | 0.828 | 0.010 | 0.001 | 0.024 | 0.999 |
| A | ua5 | Soft-voting ensemble, equal weights − XGBoost, tuned | roc_auc | 0.914 | 0.909 | 0.005 | 0.002 | 0.007 | 1.000 |
| A | ua5 | Soft-voting ensemble, equal weights − CatBoost, tuned | accuracy | 0.838 | 0.840 | -0.002 | -0.010 | 0.003 | 0.288 |
| A | ua5 | Soft-voting ensemble, equal weights − CatBoost, tuned | roc_auc | 0.914 | 0.913 | 0.001 | -0.002 | 0.004 | 0.668 |
| A | ua5 | Soft-voting ensemble, tuned weights − LightGBM, tuned | accuracy | 0.837 | 0.834 | 0.003 | -0.000 | 0.007 | 0.962 |
| A | ua5 | Soft-voting ensemble, tuned weights − LightGBM, tuned | roc_auc | 0.912 | 0.910 | 0.002 | 0.000 | 0.005 | 0.975 |
| A | ua5 | Soft-voting ensemble, tuned weights − XGBoost, tuned | accuracy | 0.837 | 0.828 | 0.010 | 0.001 | 0.024 | 0.997 |
| A | ua5 | Soft-voting ensemble, tuned weights − XGBoost, tuned | roc_auc | 0.912 | 0.909 | 0.004 | 0.001 | 0.006 | 1.000 |
| A | ua5 | Soft-voting ensemble, tuned weights − CatBoost, tuned | accuracy | 0.837 | 0.840 | -0.003 | -0.011 | 0.003 | 0.239 |
| A | ua5 | Soft-voting ensemble, tuned weights − CatBoost, tuned | roc_auc | 0.912 | 0.913 | -0.001 | -0.004 | 0.003 | 0.403 |
| B | loco | Logistic regression, tuned − LightGBM, fixed | accuracy | 0.659 | 0.705 | -0.046 | -0.079 | -0.021 | 0.000 |
| B | loco | Logistic regression, tuned − LightGBM, fixed | roc_auc | 0.701 | 0.777 | -0.077 | -0.132 | -0.030 | 0.000 |
| B | loco | Logistic regression, tuned (τ_F1) − LightGBM, fixed | f1 | 0.688 | 0.725 | -0.037 | -0.086 | -0.003 | 0.011 |
| B | loco | LightGBM, tuned − LightGBM, fixed | accuracy | 0.703 | 0.705 | -0.002 | -0.009 | 0.005 | 0.258 |
| B | loco | LightGBM, tuned − LightGBM, fixed | roc_auc | 0.776 | 0.777 | -0.001 | -0.005 | 0.002 | 0.230 |
| B | loco | LightGBM, tuned (τ_F1) − LightGBM, fixed | f1 | 0.734 | 0.725 | 0.008 | 0.001 | 0.017 | 0.993 |
| B | loco | XGBoost, tuned − LightGBM, fixed | accuracy | 0.703 | 0.705 | -0.002 | -0.007 | 0.003 | 0.226 |
| B | loco | XGBoost, tuned − LightGBM, fixed | roc_auc | 0.777 | 0.777 | -0.001 | -0.003 | 0.003 | 0.362 |
| B | loco | XGBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.734 | 0.725 | 0.008 | 0.002 | 0.017 | 0.996 |
| B | loco | CatBoost, tuned − LightGBM, fixed | accuracy | 0.704 | 0.705 | -0.001 | -0.008 | 0.006 | 0.417 |
| B | loco | CatBoost, tuned − LightGBM, fixed | roc_auc | 0.776 | 0.777 | -0.001 | -0.007 | 0.004 | 0.302 |
| B | loco | CatBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.735 | 0.725 | 0.009 | 0.001 | 0.020 | 0.993 |
| B | loco | Soft-voting ensemble, equal weights − LightGBM, fixed | accuracy | 0.706 | 0.705 | 0.001 | -0.003 | 0.006 | 0.731 |
| B | loco | Soft-voting ensemble, equal weights − LightGBM, fixed | roc_auc | 0.779 | 0.777 | 0.002 | -0.001 | 0.005 | 0.871 |
| B | loco | Soft-voting ensemble, equal weights (τ_F1) − LightGBM, fixed | f1 | 0.737 | 0.725 | 0.011 | 0.003 | 0.022 | 1.000 |
| B | loco | Soft-voting ensemble, tuned weights − LightGBM, fixed | accuracy | 0.706 | 0.705 | 0.001 | -0.004 | 0.007 | 0.700 |
| B | loco | Soft-voting ensemble, tuned weights − LightGBM, fixed | roc_auc | 0.779 | 0.777 | 0.002 | -0.001 | 0.005 | 0.854 |
| B | loco | Soft-voting ensemble, tuned weights (τ_F1) − LightGBM, fixed | f1 | 0.736 | 0.725 | 0.011 | 0.003 | 0.021 | 1.000 |
| B | loco | Soft-voting ensemble, equal weights − LightGBM, tuned | accuracy | 0.706 | 0.703 | 0.004 | 0.000 | 0.007 | 0.978 |
| B | loco | Soft-voting ensemble, equal weights − LightGBM, tuned | roc_auc | 0.779 | 0.776 | 0.003 | 0.000 | 0.006 | 0.990 |
| B | loco | Soft-voting ensemble, equal weights − XGBoost, tuned | accuracy | 0.706 | 0.703 | 0.003 | 0.001 | 0.006 | 0.996 |
| B | loco | Soft-voting ensemble, equal weights − XGBoost, tuned | roc_auc | 0.779 | 0.777 | 0.002 | 0.000 | 0.004 | 0.981 |
| B | loco | Soft-voting ensemble, equal weights − CatBoost, tuned | accuracy | 0.706 | 0.704 | 0.002 | -0.002 | 0.007 | 0.797 |
| B | loco | Soft-voting ensemble, equal weights − CatBoost, tuned | roc_auc | 0.779 | 0.776 | 0.003 | -0.000 | 0.007 | 0.962 |
| B | loco | Soft-voting ensemble, tuned weights − LightGBM, tuned | accuracy | 0.706 | 0.703 | 0.004 | 0.000 | 0.007 | 0.986 |
| B | loco | Soft-voting ensemble, tuned weights − LightGBM, tuned | roc_auc | 0.779 | 0.776 | 0.003 | -0.000 | 0.006 | 0.970 |
| B | loco | Soft-voting ensemble, tuned weights − XGBoost, tuned | accuracy | 0.706 | 0.703 | 0.003 | 0.001 | 0.006 | 0.996 |
| B | loco | Soft-voting ensemble, tuned weights − XGBoost, tuned | roc_auc | 0.779 | 0.777 | 0.002 | -0.000 | 0.004 | 0.972 |
| B | loco | Soft-voting ensemble, tuned weights − CatBoost, tuned | accuracy | 0.706 | 0.704 | 0.002 | -0.003 | 0.008 | 0.788 |
| B | loco | Soft-voting ensemble, tuned weights − CatBoost, tuned | roc_auc | 0.779 | 0.776 | 0.003 | -0.000 | 0.007 | 0.952 |
| B | ua5 | Logistic regression, tuned − LightGBM, fixed | accuracy | 0.659 | 0.698 | -0.040 | -0.074 | -0.012 | 0.001 |
| B | ua5 | Logistic regression, tuned − LightGBM, fixed | roc_auc | 0.700 | 0.772 | -0.072 | -0.128 | -0.024 | 0.001 |
| B | ua5 | Logistic regression, tuned (τ_F1) − LightGBM, fixed | f1 | 0.687 | 0.719 | -0.032 | -0.082 | 0.003 | 0.039 |
| B | ua5 | LightGBM, tuned − LightGBM, fixed | accuracy | 0.700 | 0.698 | 0.001 | -0.005 | 0.007 | 0.660 |
| B | ua5 | LightGBM, tuned − LightGBM, fixed | roc_auc | 0.772 | 0.772 | 0.001 | -0.003 | 0.005 | 0.665 |
| B | ua5 | LightGBM, tuned (τ_F1) − LightGBM, fixed | f1 | 0.732 | 0.719 | 0.013 | 0.003 | 0.025 | 0.999 |
| B | ua5 | XGBoost, tuned − LightGBM, fixed | accuracy | 0.699 | 0.698 | 0.000 | -0.005 | 0.005 | 0.562 |
| B | ua5 | XGBoost, tuned − LightGBM, fixed | roc_auc | 0.770 | 0.772 | -0.001 | -0.004 | 0.002 | 0.289 |
| B | ua5 | XGBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.729 | 0.719 | 0.010 | 0.003 | 0.017 | 1.000 |
| B | ua5 | CatBoost, tuned − LightGBM, fixed | accuracy | 0.699 | 0.698 | 0.000 | -0.005 | 0.005 | 0.553 |
| B | ua5 | CatBoost, tuned − LightGBM, fixed | roc_auc | 0.769 | 0.772 | -0.002 | -0.007 | 0.003 | 0.195 |
| B | ua5 | CatBoost, tuned (τ_F1) − LightGBM, fixed | f1 | 0.729 | 0.719 | 0.010 | 0.003 | 0.017 | 0.999 |
| B | ua5 | Soft-voting ensemble, equal weights − LightGBM, fixed | accuracy | 0.702 | 0.698 | 0.004 | -0.001 | 0.009 | 0.943 |
| B | ua5 | Soft-voting ensemble, equal weights − LightGBM, fixed | roc_auc | 0.774 | 0.772 | 0.002 | -0.001 | 0.005 | 0.914 |
| B | ua5 | Soft-voting ensemble, equal weights (τ_F1) − LightGBM, fixed | f1 | 0.732 | 0.719 | 0.013 | 0.004 | 0.021 | 1.000 |
| B | ua5 | Soft-voting ensemble, tuned weights − LightGBM, fixed | accuracy | 0.701 | 0.698 | 0.003 | -0.002 | 0.008 | 0.870 |
| B | ua5 | Soft-voting ensemble, tuned weights − LightGBM, fixed | roc_auc | 0.773 | 0.772 | 0.002 | -0.001 | 0.005 | 0.872 |
| B | ua5 | Soft-voting ensemble, tuned weights (τ_F1) − LightGBM, fixed | f1 | 0.733 | 0.719 | 0.013 | 0.006 | 0.021 | 1.000 |
| B | ua5 | Soft-voting ensemble, equal weights − LightGBM, tuned | accuracy | 0.702 | 0.700 | 0.003 | -0.001 | 0.006 | 0.943 |
| B | ua5 | Soft-voting ensemble, equal weights − LightGBM, tuned | roc_auc | 0.774 | 0.772 | 0.001 | -0.002 | 0.004 | 0.815 |
| B | ua5 | Soft-voting ensemble, equal weights − XGBoost, tuned | accuracy | 0.702 | 0.699 | 0.004 | 0.001 | 0.006 | 0.996 |
| B | ua5 | Soft-voting ensemble, equal weights − XGBoost, tuned | roc_auc | 0.774 | 0.770 | 0.003 | 0.001 | 0.006 | 0.997 |
| B | ua5 | Soft-voting ensemble, equal weights − CatBoost, tuned | accuracy | 0.702 | 0.699 | 0.004 | 0.001 | 0.007 | 0.990 |
| B | ua5 | Soft-voting ensemble, equal weights − CatBoost, tuned | roc_auc | 0.774 | 0.769 | 0.004 | 0.002 | 0.008 | 1.000 |
| B | ua5 | Soft-voting ensemble, tuned weights − LightGBM, tuned | accuracy | 0.701 | 0.700 | 0.002 | -0.001 | 0.005 | 0.869 |
| B | ua5 | Soft-voting ensemble, tuned weights − LightGBM, tuned | roc_auc | 0.773 | 0.772 | 0.001 | -0.002 | 0.004 | 0.768 |
| B | ua5 | Soft-voting ensemble, tuned weights − XGBoost, tuned | accuracy | 0.701 | 0.699 | 0.003 | -0.000 | 0.005 | 0.971 |
| B | ua5 | Soft-voting ensemble, tuned weights − XGBoost, tuned | roc_auc | 0.773 | 0.770 | 0.003 | 0.000 | 0.006 | 0.980 |
| B | ua5 | Soft-voting ensemble, tuned weights − CatBoost, tuned | accuracy | 0.701 | 0.699 | 0.003 | -0.001 | 0.007 | 0.945 |
| B | ua5 | Soft-voting ensemble, tuned weights − CatBoost, tuned | roc_auc | 0.773 | 0.769 | 0.004 | 0.001 | 0.007 | 1.000 |
