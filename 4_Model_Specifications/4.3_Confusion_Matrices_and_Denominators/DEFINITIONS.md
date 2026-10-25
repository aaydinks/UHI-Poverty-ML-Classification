# Confusion-matrix terms and denominators

Every rate in this repository states its denominator. With the median split the two classes
are almost equal in size, so a rate computed over the wrong denominator can be off by a
factor of about two without looking implausible.

| Term | Definition | Denominator |
|---|---|---|
| TP, FP, TN, FN | Raw counts of true positives, false positives, true negatives, false negatives | – |
| TPR / recall / sensitivity | TP / (TP + FN) | all actual positives |
| FNR | FN / (TP + FN) | all actual positives |
| TNR / specificity | TN / (TN + FP) | all actual negatives |
| FPR | FP / (TN + FP) | all actual negatives |
| PPV / precision | TP / (TP + FP) | all predicted positives |
| NPV | TN / (TN + FN) | all predicted negatives |
| Accuracy | (TP + TN) / n | all tracts |
| Balanced accuracy | (TPR + TNR) / 2 | – |
| F1 | 2TP / (2TP + FP + FN) | – |
| MCC | (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)) | – |

TPR + FNR = 1 and TNR + FPR = 1 (each pair shares a row denominator). PPV and NPV share no
denominator with the row rates.

Every confusion matrix is reported as counts together with rates within the actual class
(TPR, FNR, TNR, FPR), with PPV and NPV given separately over their own denominators
([`results/confusion_matrices.md`](results/confusion_matrices.md)).
