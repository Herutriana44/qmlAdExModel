"""
VQC/VQR hanya-QML dengan Angle Encoding (PauliFeatureMap).
Praproses: train_test_split → StandardScaler → PCA (denc sesuai tabel studi).
Jalankan: python qml_angle_pipeline.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit.library import PauliFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms.regressors import VQR
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from qml_study_common import (
    PipelineTrainEvalResult,
    load_classification_study_specs,
    load_regression_study_specs,
    result_to_evaluation_row,
    split_scale_pca,
    write_evaluation_csvs,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

TaskType = Literal["classification", "regression"]


def angle_feature_map(n_features: int, *, reps: int = 2) -> PauliFeatureMap:
    return PauliFeatureMap(
        feature_dimension=n_features,
        reps=reps,
        entanglement="linear",
    )


def qml_angle_train_eval(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: TaskType,
    n_pca: int,
    test_size: float = 0.25,
    random_state: int = 42,
    maxiter: int = 40,
    ansatz_reps: int = 1,
    feature_map_reps: int = 2,
) -> PipelineTrainEvalResult:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    X_tr_p, X_te_p, y_tr, y_te, scaler_x, pca = split_scale_pca(
        X,
        y,
        n_pca,
        test_size=test_size,
        random_state=random_state,
        stratify_labels=(task == "classification"),
    )

    d = X_tr_p.shape[1]
    feature_map = angle_feature_map(d, reps=feature_map_reps)
    ansatz = RealAmplitudes(
        feature_map.num_qubits, entanglement="linear", reps=ansatz_reps
    )
    optimizer = COBYLA(maxiter=maxiter)

    if task == "classification":
        label_encoder = LabelEncoder()
        y_tr = label_encoder.fit_transform(np.asarray(y_tr).ravel())
        y_te = label_encoder.transform(np.asarray(y_te).ravel())

        model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(X_tr_p, y_tr)
        y_pred = np.asarray(model.predict(X_te_p))
        acc = float(accuracy_score(y_te, y_pred))
        prec = float(
            precision_score(y_te, y_pred, average="macro", zero_division=0)
        )
        rec = float(recall_score(y_te, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        metrics = {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
        }
        print(f"Test accuracy: {acc:.4f}")
        print(classification_report(y_te, y_pred, digits=4))

        return PipelineTrainEvalResult(
            task=task,
            model=model,
            scaler_x=scaler_x,
            pca=pca,
            scaler_z=None,
            label_encoder=label_encoder,
            scaler_y=None,
            metrics=metrics,
            y_test=y_te,
            y_pred=y_pred,
            n_samples=int(X.shape[0]),
        )

    if task == "regression":
        y_tr = np.asarray(y_tr, dtype=float).ravel()
        y_te = np.asarray(y_te, dtype=float).ravel()
        scaler_y = StandardScaler()
        y_trs = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

        model = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(X_tr_p, y_trs)
        y_pred_s = np.asarray(model.predict(X_te_p), dtype=float).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_te, y_pred)
        mse = mean_squared_error(y_te, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_te, y_pred)
        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": rmse,
            "r2": float(r2),
        }
        print(f"Test MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")

        return PipelineTrainEvalResult(
            task=task,
            model=model,
            scaler_x=scaler_x,
            pca=pca,
            scaler_z=None,
            label_encoder=None,
            scaler_y=scaler_y,
            metrics=metrics,
            y_test=y_te,
            y_pred=y_pred,
            n_samples=int(X.shape[0]),
        )

    raise ValueError("task must be 'classification' or 'regression'")


def main():
    parser = argparse.ArgumentParser(description="Angle Encoding QML Pipeline")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "both"],
        default="both",
        help="Task to run: classification, regression, or both (default: both)",
    )
    args = parser.parse_args()

    test_size = 0.25
    random_state = 42
    clf_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []

    if args.task in ["classification", "both"]:
        for spec in load_classification_study_specs():
            print(f"\n=== {spec.title} klasifikasi — Angle (Pauli), denc={spec.denc} ===")
            res = qml_angle_train_eval(
                spec.X,
                spec.y,
                task="classification",
                n_pca=spec.denc,
                maxiter=spec.maxiter,
                test_size=test_size,
                random_state=random_state,
            )
            clf_rows.append(
                result_to_evaluation_row(
                    res,
                    spec.csv_name,
                    spec.d_orig,
                    spec.denc,
                    spec.maxiter,
                    test_size,
                    random_state,
                )
            )

    if args.task in ["regression", "both"]:
        for spec in load_regression_study_specs():
            print(f"\n=== {spec.title} (regresi, Angle) ===")
            d_orig = spec.X.shape[1]
            n_pca = 8 if d_orig >= 8 else d_orig
            res = qml_angle_train_eval(
                spec.X,
                spec.y,
                task="regression",
                n_pca=n_pca,
                maxiter=spec.maxiter,
                test_size=test_size,
                random_state=random_state,
            )
            reg_rows.append(
                result_to_evaluation_row(
                    res,
                    spec.csv_name,
                    spec.d_orig,
                    n_pca,
                    spec.maxiter,
                    test_size,
                    random_state,
                )
            )

    out_dir = Path(__file__).resolve().parent
    clf_path, reg_path = write_evaluation_csvs(
        clf_rows,
        reg_rows,
        out_dir,
        clf_filename="eval_classification_angle.csv",
        reg_filename="eval_regression_angle.csv",
    )
    print("\n=== Ekspor evaluasi (CSV) — Angle ===")
    if args.task in ["classification", "both"]:
        print(f"  Klasifikasi: {clf_path}")
    if args.task in ["regression", "both"]:
        print(f"  Regresi:     {reg_path}")


if __name__ == "__main__":
    main()
