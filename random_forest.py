"""Train a Random Forest on BEED train/test CSVs and print WEKA-style metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
ORIGINAL_PATH = ROOT / "BEED_Data.csv"
TRAIN_PATH = ROOT / "BEED_Data_train.csv"
TEST_PATH = ROOT / "BEED_Data_test.csv"
FEATURES = [f"X{i}" for i in range(1, 17)]
TARGET = "y"
RANDOM_STATE = 42


def load_xy(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def weka_error_stats(y_true, y_proba, classes, y_train) -> dict[str, float]:
    """Match WEKA MAE/RMSE using class probability vs one-hot labels."""
    class_index = {label: i for i, label in enumerate(classes)}
    n_classes = len(classes)
    n = len(y_true)

    actual = np.zeros((n, n_classes))
    for i, label in enumerate(y_true):
        actual[i, class_index[label]] = 1.0

    delta = y_proba - actual
    n_terms = n * n_classes
    mae = np.abs(delta).sum() / n_terms
    rmse = np.sqrt((delta**2).sum() / n_terms)

    priors = np.array([(y_train == label).mean() for label in classes])
    prior_delta = np.tile(priors, (n, 1)) - actual
    mae_prior = np.abs(prior_delta).sum() / n_terms
    rmse_prior = np.sqrt((prior_delta**2).sum() / n_terms)

    return {
        "mae": mae,
        "rmse": rmse,
        "rae": 100.0 * mae / mae_prior,
        "rrse": 100.0 * rmse / rmse_prior,
    }


def per_class_fp_rate(matrix: np.ndarray) -> np.ndarray:
    fp = matrix.sum(axis=0) - np.diag(matrix)
    tn = matrix.sum() - (matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix))
    return fp / np.clip(fp + tn, 1, None)


def print_weka_summary(y_true, y_pred, y_proba, classes, y_train) -> None:
    n = len(y_true)
    correct = int((y_pred == y_true).sum())
    incorrect = n - correct
    accuracy_pct = 100.0 * correct / n
    error_pct = 100.0 * incorrect / n
    kappa = cohen_kappa_score(y_true, y_pred)
    errors = weka_error_stats(y_true, y_proba, classes, y_train)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)

    precision, recall, fmeasure, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    fp_rate = per_class_fp_rate(matrix)

    y_bin = np.zeros((n, len(classes)), dtype=int)
    for i, label in enumerate(classes):
        y_bin[:, i] = (y_true == label).astype(int)

    mcc = np.array(
        [matthews_corrcoef(y_bin[:, i], (y_pred == label).astype(int)) for i, label in enumerate(classes)]
    )
    roc = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None, labels=classes)
    prc = np.array(
        [average_precision_score(y_bin[:, i], y_proba[:, i]) for i in range(len(classes))]
    )

    weights = support / support.sum()
    weighted = {
        "tp": np.average(recall, weights=weights),
        "fp": np.average(fp_rate, weights=weights),
        "prec": np.average(precision, weights=weights),
        "rec": np.average(recall, weights=weights),
        "f": np.average(fmeasure, weights=weights),
        "mcc": np.average(mcc, weights=weights),
        "roc": np.average(roc, weights=weights),
        "prc": np.average(prc, weights=weights),
    }

    print("=== Summary ===")
    print()
    print(f"Correctly Classified Instances        {correct:5d}              {accuracy_pct:6.4g} %")
    print(f"Incorrectly Classified Instances      {incorrect:5d}              {error_pct:6.4g} %")
    print(f"Kappa statistic                          {kappa:.4f}")
    print(f"Mean absolute error                      {errors['mae']:.4f}")
    print(f"Root mean squared error                  {errors['rmse']:.4f}")
    print(f"Relative absolute error                  {errors['rae']:.4f} %")
    print(f"Root relative squared error              {errors['rrse']:.4f} %")
    print(f"Total Number of Instances             {n:5d}")
    print()
    print("=== Detailed Accuracy By Class ===")
    print()
    header = (
        "                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      "
        "ROC Area  PRC Area  Class"
    )
    print(header)
    for i, label in enumerate(classes):
        print(
            f"                 {recall[i]:7.3f}  {fp_rate[i]:7.3f}  {precision[i]:9.3f}  "
            f"{recall[i]:7.3f}  {fmeasure[i]:9.3f}  {mcc[i]:7.3f}  {roc[i]:8.3f}  "
            f"{prc[i]:8.3f}  {label}"
        )
    print(
        f"Weighted Avg.    {weighted['tp']:7.3f}  {weighted['fp']:7.3f}  {weighted['prec']:9.3f}  "
        f"{weighted['rec']:7.3f}  {weighted['f']:9.3f}  {weighted['mcc']:7.3f}  "
        f"{weighted['roc']:8.3f}  {weighted['prc']:8.3f}"
    )
    print()
    print("=== Confusion Matrix ===")
    print()
    letters = [chr(ord("a") + i) for i in range(len(classes))]
    print("    " + "  ".join(f"{letter:>3}" for letter in letters) + "   <-- classified as")
    width = 5
    for i, label in enumerate(classes):
        row = "".join(f"{int(v):{width}d}" for v in matrix[i])
        print(f"{row} |  {letters[i]} = {label}")


def main() -> None:
    original = pd.read_csv(ORIGINAL_PATH)
    print(f"Scheme:       sklearn.ensemble.RandomForestClassifier")
    print(f"Relation:     {ORIGINAL_PATH.name} ({len(original)} instances)")
    print(f"Instances:    training {TRAIN_PATH.name} / testing {TEST_PATH.name}")
    print()

    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print_weka_summary(y_test.to_numpy(), y_pred, y_proba, model.classes_, y_train.to_numpy())


if __name__ == "__main__":
    main()
