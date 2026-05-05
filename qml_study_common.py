"""
Konfigurasi dataset dan praproses bersama untuk studi encoding (tabel PCA / denc).
Praproses: train_test_split → StandardScaler (fit pada train) → PCA (fit pada train).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)

STUDY_RANDOM_STATE = 42

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class ClassificationStudySpec:
    """Satu baris studi klasifikasi."""

    csv_name: str
    title: str
    X: np.ndarray
    y: np.ndarray
    n_samples: int
    d_orig: int
    n_classes: int
    denc: int
    maxiter: int


@dataclass(frozen=True)
class RegressionStudySpec:
    """Satu baris studi regresi."""

    csv_name: str
    title: str
    X: np.ndarray
    y: np.ndarray
    n_samples: int
    d_orig: int
    denc: int
    maxiter: int


def load_classification_study_specs() -> list[ClassificationStudySpec]:
    """
    Memuat 7 dataset klasifikasi sesuai tabel studi.
    """
    rs = STUDY_RANDOM_STATE
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()
    Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=rs)
    Xc, yc = make_circles(
        n_samples=300, noise=0.05, factor=0.5, random_state=rs
    )
    Xb, yb = make_blobs(
        n_samples=300,
        centers=3,
        n_features=5,
        cluster_std=1.0,
        random_state=rs,
    )

    specs: list[tuple[ClassificationStudySpec, tuple[int, int]]] = [
        (
            ClassificationStudySpec(
                csv_name="iris_species",
                title="Iris",
                X=np.asarray(iris.data, dtype=float),
                y=np.asarray(iris.target),
                n_samples=150,
                d_orig=4,
                n_classes=3,
                denc=4,
                maxiter=80,
            ),
            iris.data.shape,
        ),
        (
            ClassificationStudySpec(
                csv_name="wine",
                title="Wine",
                X=np.asarray(wine.data, dtype=float),
                y=np.asarray(wine.target),
                n_samples=178,
                d_orig=13,
                n_classes=3,
                denc=8,
                maxiter=70,
            ),
            wine.data.shape,
        ),
        (
            ClassificationStudySpec(
                csv_name="breast_cancer",
                title="Breast cancer",
                X=np.asarray(cancer.data, dtype=float),
                y=np.asarray(cancer.target),
                n_samples=569,
                d_orig=30,
                n_classes=2,
                denc=8,
                maxiter=60,
            ),
            cancer.data.shape,
        ),
        (
            ClassificationStudySpec(
                csv_name="digits",
                title="Digits",
                X=np.asarray(digits.data, dtype=float),
                y=np.asarray(digits.target),
                n_samples=1797,
                d_orig=64,
                n_classes=10,
                denc=8,
                maxiter=50,
            ),
            digits.data.shape,
        ),
        (
            ClassificationStudySpec(
                csv_name="moons",
                title="Moons",
                X=np.asarray(Xm, dtype=float),
                y=np.asarray(ym),
                n_samples=300,
                d_orig=2,
                n_classes=2,
                denc=2,
                maxiter=70,
            ),
            Xm.shape,
        ),
        (
            ClassificationStudySpec(
                csv_name="circles",
                title="Circles",
                X=np.asarray(Xc, dtype=float),
                y=np.asarray(yc),
                n_samples=300,
                d_orig=2,
                n_classes=2,
                denc=2,
                maxiter=70,
            ),
            Xc.shape,
        ),
        (
            ClassificationStudySpec(
                csv_name="blobs",
                title="Blobs",
                X=np.asarray(Xb, dtype=float),
                y=np.asarray(yb),
                n_samples=300,
                d_orig=5,
                n_classes=3,
                denc=5,
                maxiter=70,
            ),
            Xb.shape,
        ),
    ]

    out: list[ClassificationStudySpec] = []
    for spec, shape in specs:
        if shape[0] != spec.n_samples or shape[1] != spec.d_orig:
            raise ValueError(
                f"{spec.title}: bentuk data {shape} tidak cocok tabel "
                f"(n={spec.n_samples}, d={spec.d_orig})."
            )
        n_cls = int(len(np.unique(spec.y)))
        if n_cls != spec.n_classes:
            raise ValueError(
                f"{spec.title}: #kelas {n_cls} != {spec.n_classes}."
            )
        out.append(spec)
    return out


def load_regression_study_specs() -> list[RegressionStudySpec]:
    """
    Memuat 3 dataset regresi sesuai pipeline.
    """
    iris = load_iris()
    dia = load_diabetes()
    cal = fetch_california_housing()

    return [
        RegressionStudySpec(
            csv_name="iris_petal_width",
            title="Iris Petal Width",
            X=np.asarray(iris.data[:, :3], dtype=float),
            y=np.asarray(iris.data[:, 3], dtype=float),
            n_samples=150,
            d_orig=3,
            denc=3,
            maxiter=80,
        ),
        RegressionStudySpec(
            csv_name="diabetes_progression",
            title="Diabetes",
            X=np.asarray(dia.data, dtype=float),
            y=np.asarray(dia.target, dtype=float),
            n_samples=442,
            d_orig=10,
            denc=8,
            maxiter=60,
        ),
        RegressionStudySpec(
            csv_name="california_housing",
            title="California Housing",
            X=np.asarray(cal.data, dtype=float),
            y=np.asarray(cal.target, dtype=float),
            n_samples=20640,
            d_orig=8,
            denc=8,
            maxiter=50,
        ),
    ]



@dataclass
class PipelineTrainEvalResult:
    task: TaskType
    model: Any
    scaler_x: StandardScaler
    pca: PCA
    scaler_z: StandardScaler | None
    label_encoder: Any | None
    scaler_y: StandardScaler | None
    metrics: dict[str, float]
    y_test: np.ndarray
    y_pred: np.ndarray
    n_samples: int


CLASSIFICATION_CSV_FIELDS = [
    "dataset",
    "task",
    "n_samples",
    "n_test",
    "d_orig",
    "denc",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "maxiter",
    "test_size",
    "random_state",
]

REGRESSION_CSV_FIELDS = [
    "dataset",
    "task",
    "n_samples",
    "n_test",
    "d_orig",
    "denc",
    "mae",
    "mse",
    "rmse",
    "r2",
    "maxiter",
    "test_size",
    "random_state",
]


def result_to_evaluation_row(
    result: PipelineTrainEvalResult,
    dataset: str,
    d_orig: int,
    denc: int,
    maxiter: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    n_test = int(len(result.y_test))
    row: dict[str, Any] = {
        "dataset": dataset,
        "task": result.task,
        "n_samples": result.n_samples,
        "n_test": n_test,
        "d_orig": d_orig,
        "denc": denc,
        "maxiter": maxiter,
        "test_size": test_size,
        "random_state": random_state,
    }
    if result.task == "classification":
        row["accuracy"] = result.metrics["accuracy"]
        row["precision_macro"] = result.metrics["precision_macro"]
        row["recall_macro"] = result.metrics["recall_macro"]
        row["f1_macro"] = result.metrics["f1_macro"]
    else:
        row["mae"] = result.metrics["mae"]
        row["mse"] = result.metrics["mse"]
        row["rmse"] = result.metrics["rmse"]
        row["r2"] = result.metrics["r2"]
    return row


def write_evaluation_csvs(
    classification_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    output_dir: str | Path = ".",
    *,
    clf_filename: str = "eval_classification.csv",
    reg_filename: str = "eval_regression.csv",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clf_path = output_dir / clf_filename
    reg_path = output_dir / reg_filename

    for path, rows, fields in (
        (clf_path, classification_rows, CLASSIFICATION_CSV_FIELDS),
        (reg_path, regression_rows, REGRESSION_CSV_FIELDS),
    ):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    return clf_path, reg_path


def split_scale_pca(
    X: np.ndarray,
    y: np.ndarray,
    n_pca: int,
    *,
    test_size: float,
    random_state: int,
    stratify_labels: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    PCA,
]:
    """
    Split lalu StandardScaler (train-only) lalu PCA (train-only).
    n_pca dipotong ke min(n_pca, n_fitur_latih).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    strat = y if stratify_labels else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    n_comp = min(int(n_pca), X_tr_s.shape[1])
    pca = PCA(n_components=n_comp)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    return X_tr_p, X_te_p, y_tr, y_te, scaler, pca


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms
