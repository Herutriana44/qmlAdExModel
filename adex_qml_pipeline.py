"""
AdEx (Brian2) encoding + Qiskit VQC/VQR — klasifikasi & regresi.
Praproses: split → StandardScaler → PCA (denc) → AdEx → VQC/VQR.
Dataset klasifikasi mengikuti tabel studi (lihat `qml_study_common.load_classification_study_specs`).
Jalankan: python adex_qml_pipeline.py
Atau impor fungsi `adex_qml_train_eval` dari notebook.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
from brian2 import *
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms.regressors import VQR
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

# Nama lama untuk kompatibilitas impor
AdExQMLResult = PipelineTrainEvalResult


def run_adex_vectorized(x, noise=None):
    """Satu sampel: vektor arus panjang N → N neuron AdEx."""
    start_scope()
    defaultclock.dt = 0.01 * ms

    x = np.asarray(x, dtype=float).ravel()
    if noise is None:
        noise = np.zeros_like(x)
    noise = np.asarray(noise, dtype=float).ravel()
    assert len(x) == len(noise)

    C = 281 * pF
    gL = 30 * nS
    EL = -70.6 * mV
    VT = -50.4 * mV
    DeltaT = 2 * mV
    tauw = 40 * ms
    a = 4 * nS
    b = 0.08 * nA
    Vcut = VT + 5 * DeltaT

    N = len(x)
    duration = 1 * second
    init_time = 3 * second
    bin_size = 10 * ms
    num_bins = int(duration / bin_size)

    I_array = (x + noise) * nA

    eqs = """
    dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I-w)/C : volt
    dw/dt=(a*(vm-EL)-w)/tauw : amp
    I : amp
    Vr:volt
    """

    neuron = NeuronGroup(
        N,
        model=eqs,
        threshold="vm > Vcut",
        reset="vm = Vr; w += b",
        method="euler",
    )
    neuron.vm = EL
    neuron.w = a * (neuron.vm - EL)
    neuron.Vr = linspace(-48.3 * mV, -47.7 * mV, N)
    neuron.I = I_array

    run(init_time)
    spikes = SpikeMonitor(neuron)
    states = StateMonitor(neuron, ["w"], record=True)
    run(duration)

    times = (spikes.t - init_time) / second
    time_array = (states.t - init_time) / second
    firing_rate = spikes.count / duration

    binned_spikes = np.zeros((N, num_bins))
    for idx, t in zip(spikes.i, times):
        if t >= 0:
            b = int(t / (bin_size / second))
            if 0 <= b < num_bins:
                binned_spikes[idx, b] += 1

    w_data = states.w / nA
    binned_w = np.zeros((N, num_bins))
    counts = np.zeros((N, num_bins))
    bin_indices = (time_array / (bin_size / second)).astype(int)
    for t_idx, b in enumerate(bin_indices):
        if 0 <= b < num_bins and time_array[t_idx] >= 0:
            binned_w[:, b] += w_data[:, t_idx]
            counts[:, b] += 1
    mask = counts > 0
    binned_w[mask] /= counts[mask]

    latest_firing_rate = firing_rate
    latest_spike = binned_spikes[:, -1]
    latest_w = binned_w[:, -1]

    latest_spike_time = np.full(N, np.nan)
    for i, t in zip(spikes.i, times):
        if t >= 0:
            latest_spike_time[i] = t * 1000

    return latest_firing_rate, latest_spike_time, latest_spike, latest_w


def nan_to_num(array):
    return np.where(np.isnan(array), 0.0, array)


def adex_style_feature_map(n_features: int) -> QuantumCircuit:
    x_params = ParameterVector("x", n_features)
    qc = QuantumCircuit(2 * n_features)
    for i in range(n_features):
        qc.rx(x_params[i], i)
        qc.rz(x_params[i], i)
    for j in range(n_features, 2 * n_features):
        qc.h(j)
    qc.barrier()
    for i in range(n_features):
        for j in range(n_features, 2 * n_features):
            qc.cx(j, i)
            qc.rz(x_params[i], j)
            qc.cx(i, j)
    qc.barrier()
    return qc


def _encode_row_adex(x_row: np.ndarray) -> np.ndarray:
    fr, st, sp, w = run_adex_vectorized(x_row)
    fr = np.asarray(fr, dtype=float)
    st = nan_to_num(np.asarray(st, dtype=float))
    sp = np.asarray(sp, dtype=float)
    w = np.asarray(w, dtype=float)
    return np.array(
        [
            float(np.mean(fr)),
            float(np.mean(st)),
            float(np.mean(sp)),
            float(np.mean(w)),
        ],
        dtype=float,
    )


def encode_dataset_adex(X: np.ndarray, verbose: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = np.zeros((X.shape[0], 4), dtype=float)
    for i in range(X.shape[0]):
        if verbose and (i % 20 == 0 or i == X.shape[0] - 1):
            print(f"  AdEx encoding: {i + 1}/{X.shape[0]}")
        out[i] = _encode_row_adex(X[i])
    return out


def adex_qml_train_eval(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: TaskType,
    n_pca: int,
    test_size: float = 0.25,
    random_state: int = 42,
    maxiter: int = 40,
    ansatz_reps: int = 1,
    verbose_adex: bool = True,
) -> PipelineTrainEvalResult:
    """
    Pipeline: split → StandardScaler → PCA (n_pca) → AdEx (4 ringkasan) →
    StandardScaler(Z) → VQC / VQR.
    """
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

    print("Encoding AdEx (Brian2)…")
    Z_tr = encode_dataset_adex(X_tr_p, verbose=verbose_adex)
    Z_te = encode_dataset_adex(X_te_p, verbose=verbose_adex)

    n_features = Z_tr.shape[1]
    feature_map = adex_style_feature_map(n_features)
    ansatz = RealAmplitudes(2 * n_features, entanglement="linear", reps=ansatz_reps)
    optimizer = COBYLA(maxiter=maxiter)

    if task == "classification":
        label_encoder = LabelEncoder()
        y_tr = label_encoder.fit_transform(np.asarray(y_tr).ravel())
        y_te = label_encoder.transform(np.asarray(y_te).ravel())
        scaler_z = StandardScaler()
        z_trs = scaler_z.fit_transform(Z_tr)
        z_tes = scaler_z.transform(Z_te)

        model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(z_trs, y_tr)
        y_pred = model.predict(z_tes)
        y_pred = np.asarray(y_pred)
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
            scaler_z=scaler_z,
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
        scaler_z = StandardScaler()
        scaler_y = StandardScaler()
        z_trs = scaler_z.fit_transform(Z_tr)
        z_tes = scaler_z.transform(Z_te)
        y_trs = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

        model = VQR(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(z_trs, y_trs)
        y_pred_s = np.asarray(model.predict(z_tes), dtype=float).ravel()
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
            scaler_z=scaler_z,
            label_encoder=None,
            scaler_y=scaler_y,
            metrics=metrics,
            y_test=y_te,
            y_pred=y_pred,
            n_samples=int(X.shape[0]),
        )

    raise ValueError("task must be 'classification' or 'regression'")


def main():
    parser = argparse.ArgumentParser(description="AdEx QML Pipeline")
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
            print(f"\n=== {spec.title} klasifikasi (tabel studi, denc={spec.denc}) ===")
            res_clf = adex_qml_train_eval(
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
                    res_clf,
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
            print(f"\n=== {spec.title} regresi (tabel studi, denc={spec.denc}) ===")
            res_reg = adex_qml_train_eval(
                spec.X,
                spec.y,
                task="regression",
                n_pca=spec.denc,
                maxiter=spec.maxiter,
                test_size=test_size,
                random_state=random_state,
            )
            reg_rows.append(
                result_to_evaluation_row(
                    res_reg,
                    spec.csv_name,
                    spec.d_orig,
                    spec.denc,
                    spec.maxiter,
                    test_size,
                    random_state,
                )
            )

    out_dir = Path(__file__).resolve().parent
    clf_path, reg_path = write_evaluation_csvs(clf_rows, reg_rows, out_dir)
    print("\n=== Ekspor evaluasi (CSV) ===")
    if args.task in ["classification", "both"]:
        print(f"  Klasifikasi: {clf_path}")
    if args.task in ["regression", "both"]:
        print(f"  Regresi:     {reg_path}")


if __name__ == "__main__":
    main()
