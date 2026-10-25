# 4.3 Confusion matrices and denominators

## Purpose

This section reports the full confusion matrix of every model and states what each rate is
divided by. With the median split the two classes are almost equal in size, so a rate over the
wrong denominator can be off by a factor of about two without looking implausible.

## Method

- [`DEFINITIONS.md`](DEFINITIONS.md) defines every term with its denominator.
- `confusion_matrices.py` writes, for every model × task under leave-one-city-out and
  UA-grouped 5-fold, the four counts, all six rates and each rate's denominator
  ([`results/confusion_matrices.csv`](results/confusion_matrices.csv)), and one annotated
  figure per task. Each matrix is checked to sum to n.

## Results

Leave-one-city-out, n = 54,243 tracts (τ = 0.5); rate (numerator/denominator):

| Task | Model | TP | FP | TN | FN | TPR | FPR | PPV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Logistic regression | 23,124 | 5,223 | 21,898 | 3,998 | 0.853 (23,124/27,122) | 0.193 (5,223/27,121) | 0.816 (23,124/28,347) |
| A | LightGBM | 21,721 | 4,182 | 22,939 | 5,401 | 0.801 (21,721/27,122) | 0.154 (4,182/27,121) | 0.839 (21,721/25,903) |
| B | Logistic regression | 18,082 | 9,137 | 17,984 | 9,040 | 0.667 (18,082/27,122) | 0.337 (9,137/27,121) | 0.664 (18,082/27,219) |
| B | LightGBM | 21,122 | 10,004 | 17,117 | 6,000 | 0.779 (21,122/27,122) | 0.369 (10,004/27,121) | 0.679 (21,122/31,126) |

The majority-class baseline predicts one class for every tract in a fold: under
leave-one-city-out on Task A it yields TP = 7,216, FP = 16,095, TN = 11,026,
FN = 19,906. The national median split leaves the two classes within one tract of each other,
so holding out a city tips the training majority toward the class that city has *less* of:
under leave-one-city-out the majority class predicts "positive" in 385
of 486 folds on Task A and its accuracy falls to
0.336 (0.391 on Task B;
[`results/majority_folds.md`](results/majority_folds.md)). Pooled over folds, the baseline is
therefore worse than chance, not merely equal to it. The no-skill references to compare models
against are balanced accuracy 0.5, MCC 0 and ROC-AUC 0.5.

Figures: [`results/confusion_taskA.png`](results/confusion_taskA.png),
[`results/confusion_taskB.png`](results/confusion_taskB.png). UA-grouped 5-fold matrices are
in the same CSV.
