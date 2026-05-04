"""
VQC/VQR hanya-QML dengan Amplitude Encoding (RawFeatureVector).
Praproses: split → StandardScaler → PCA (denc) → normalisasi L2 baris (state amplitudo).
Jalankan: python qml_amplitude_pipeline.py
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms.regressors import VQR
from qiskit_machine_learning.circuit.library import raw_feature_vector
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
    l2_normalize_rows,
    result_to_evaluation_row,
    split_scale_pca,
    write_evaluation_csvs,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Membungkam peringatan precision/f1 ketika ada kelas yang tidak terprediksi
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

TaskType = Literal["classification", "regression"]

def pad_to_power_of_2(x: np.ndarray) -> np.ndarray:
    """Menambah kolom nol agar jumlah fitur menjadi pangkat 2 (2, 4, 8, 16, dst)."""
    n_features = x.shape[1]
    if n_features == 0:
        return x
    target_dim = 1 << (n_features - 1).bit_length()
    if n_features == target_dim:
        return x
    padding = target_dim - n_features
    return np.hstack([x, np.zeros((x.shape[0], padding))])

def amplitude_feature_map(n_features: int) -> Any:
    return raw_feature_vector(n_features)

def qml_amplitude_train_eval(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: TaskType,
    n_pca: int,
    test_size: float = 0.25,
    random_state: int = 42,
    maxiter: int = 40,
    ansatz_reps: int = 1,
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

    # 1. Normalisasi L2 (Syarat Amplitude Encoding)
    z_tr = l2_normalize_rows(X_tr_p)
    z_te = l2_normalize_rows(X_te_p)

    # 2. FIX: Padding ke pangkat 2 terdekat (misal 5 -> 8)
    z_tr = pad_to_power_of_2(z_tr)
    z_te = pad_to_power_of_2(z_te)

    d_final = z_tr.shape[1]
    feature_map = amplitude_feature_map(d_final)

    ansatz = RealAmplitudes(
        feature_map.num_qubits, entanglement="linear", reps=ansatz_reps
    )
    optimizer = COBYLA(maxiter=maxiter)

    if task == "classification":
        label_encoder = LabelEncoder()
        y_tr = label_encoder.fit_transform(np.asarray(y_tr).ravel())
        y_te = label_encoder.transform(np.asarray(y_te).ravel())

        model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(z_tr, y_tr)
        y_pred = np.asarray(model.predict(z_te))

        metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "precision_macro": float(precision_score(y_te, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_te, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        }

        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(classification_report(y_te, y_pred, digits=4, zero_division=0))

        return PipelineTrainEvalResult(
            task=task, model=model, scaler_x=scaler_x, pca=pca,
            scaler_z=None, label_encoder=label_encoder, scaler_y=None,
            metrics=metrics, y_test=y_te, y_pred=y_pred, n_samples=int(X.shape[0]),
        )

    if task == "regression":
        y_tr = np.asarray(y_tr, dtype=float).ravel()
        y_te = np.asarray(y_te, dtype=float).ravel()
        scaler_y = StandardScaler()
        y_trs = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

        model = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(z_tr, y_trs)
        y_pred_s = np.asarray(model.predict(z_te), dtype=float).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_te, y_pred)
        rmse = float(np.sqrt(mse))
        metrics = {
            "mae": float(mean_absolute_error(y_te, y_pred)),
            "mse": float(mse),
            "rmse": rmse,
            "r2": float(r2_score(y_te, y_pred)),
        }
        print(f"Test MAE={metrics['mae']:.4f} RMSE={rmse:.4f} R²={metrics['r2']:.4f}")

        return PipelineTrainEvalResult(
            task=task, model=model, scaler_x=scaler_x, pca=pca,
            scaler_z=None, label_encoder=None, scaler_y=scaler_y,
            metrics=metrics, y_test=y_te, y_pred=y_pred, n_samples=int(X.shape[0]),
        )

    raise ValueError("task must be 'classification' or 'regression'")

def main():
    test_size = 0.25
    random_state = 42
    clf_rows, reg_rows = [], []
    iris = load_iris()

    for spec in load_classification_study_specs():
        print(f"\n=== {spec.title} klasifikasi — Amplitude (RawFeatureVector), denc={spec.denc} ===")
        print(f"jumlah fitur : {spec.X.shape[1]}")
        res = qml_amplitude_train_eval(
            spec.X, spec.y, task="classification", n_pca=spec.denc,
            maxiter=spec.maxiter, test_size=test_size, random_state=random_state,
        )
        clf_rows.append(result_to_evaluation_row(
            res, spec.csv_name, spec.d_orig, spec.denc, spec.maxiter, test_size, random_state
        ))

    # --- Regresi ---
    datasets = [
        ("iris_petal_width", iris.data[:, :3], iris.data[:, 3], 80),
        ("diabetes_progression", load_diabetes().data, load_diabetes().target, 60),
        ("california_housing", fetch_california_housing().data, fetch_california_housing().target, 50)
    ]

    for name, X, y, m_iter in datasets:
        print(f"\n=== {name} (regresi, Amplitude) ===")
        d_orig = X.shape[1]
        # Gunakan 8 atau d_orig, padding ditangani otomatis di fungsi
        n_pca = 8 if d_orig >= 8 else d_orig
        res = qml_amplitude_train_eval(
            X, y, task="regression", n_pca=n_pca,
            maxiter=m_iter, test_size=test_size, random_state=random_state,
        )
        reg_rows.append(result_to_evaluation_row(
            res, name, d_orig, n_pca, m_iter, test_size, random_state
        ))

    out_dir = Path(__file__).resolve().parent
    clf_path, reg_path = write_evaluation_csvs(
        clf_rows, reg_rows, out_dir,
        clf_filename="eval_classification_amplitude.csv",
        reg_filename="eval_regression_amplitude.csv",
    )
    print(f"\n=== Ekspor CSV SELESAI ===\nKlasifikasi: {clf_path}\nRegresi: {reg_path}")

if __name__ == "__main__":
    main()