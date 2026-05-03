"""
AdEx (Brian2) encoding + Qiskit VQC/VQR — klasifikasi & regresi.
Jalankan: python adex_qml_pipeline.py
Atau impor fungsi `adex_qml_train_eval` dari notebook.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from brian2 import *
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms.regressors import VQR
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
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)

TaskType = Literal["classification", "regression"]


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


@dataclass
class AdExQMLResult:
    task: TaskType
    model: Any
    scaler_z: StandardScaler
    scaler_y: StandardScaler | None
    label_encoder: LabelEncoder | None
    metrics: dict[str, float]
    y_test: np.ndarray
    y_pred: np.ndarray
    n_samples: int


CLASSIFICATION_CSV_FIELDS = [
    "dataset",
    "task",
    "n_samples",
    "n_test",
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
    "mae",
    "mse",
    "rmse",
    "r2",
    "maxiter",
    "test_size",
    "random_state",
]


def result_to_evaluation_row(
    result: AdExQMLResult,
    dataset: str,
    maxiter: int,
    test_size: float,
    random_state: int,
) -> dict[str, Any]:
    """Satu baris ringkasan evaluasi untuk diekspor ke CSV."""
    n_test = int(len(result.y_test))
    row: dict[str, Any] = {
        "dataset": dataset,
        "task": result.task,
        "n_samples": result.n_samples,
        "n_test": n_test,
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
) -> tuple[Path, Path]:
    """
    Menulis dua berkas: eval_classification.csv dan eval_regression.csv.
    Jika salah satu daftar kosong, berkas tetap dibuat hanya berisi header.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clf_path = output_dir / "eval_classification.csv"
    reg_path = output_dir / "eval_regression.csv"

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


def stratified_train_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_subset: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ambil subset berukuran n_subset dengan proporsi kelas tetap (untuk stratify di split berikutnya)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = min(int(n_subset), len(X))
    if n == len(X):
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=random_state)
    tr_idx, _ = next(sss.split(X, y))
    return X[tr_idx], y[tr_idx]


def adex_qml_train_eval(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: TaskType,
    test_size: float = 0.25,
    random_state: int = 42,
    maxiter: int = 40,
    ansatz_reps: int = 1,
    verbose_adex: bool = True,
) -> AdExQMLResult:
    """
    Pipeline: StandardScaler(X) → encoding AdEx (4 ringkasan) → StandardScaler(Z) →
    VQC (klasifikasi) atau VQR (regresi).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    x_scaler = StandardScaler()
    Xs = x_scaler.fit_transform(X)

    print("Encoding AdEx (Brian2)…")
    Z = encode_dataset_adex(Xs, verbose=verbose_adex)

    n_features = Z.shape[1]
    feature_map = adex_style_feature_map(n_features)
    ansatz = RealAmplitudes(2 * n_features, entanglement="linear", reps=ansatz_reps)
    optimizer = COBYLA(maxiter=maxiter)

    if task == "classification":
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y.ravel())
        z_tr, z_te, y_tr, y_te = train_test_split(
            Z, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
        )
        scaler_z = StandardScaler()
        z_trs = scaler_z.fit_transform(z_tr)
        z_tes = scaler_z.transform(z_te)

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

        return AdExQMLResult(
            task=task,
            model=model,
            scaler_z=scaler_z,
            scaler_y=None,
            label_encoder=label_encoder,
            metrics=metrics,
            y_test=y_te,
            y_pred=y_pred,
            n_samples=int(X.shape[0]),
        )

    if task == "regression":
        y_f = np.asarray(y, dtype=float).ravel()
        z_tr, z_te, y_tr, y_te = train_test_split(
            Z, y_f, test_size=test_size, random_state=random_state
        )
        scaler_z = StandardScaler()
        scaler_y = StandardScaler()
        z_trs = scaler_z.fit_transform(z_tr)
        z_tes = scaler_z.transform(z_te)
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

        return AdExQMLResult(
            task=task,
            model=model,
            scaler_z=scaler_z,
            scaler_y=scaler_y,
            label_encoder=None,
            metrics=metrics,
            y_test=y_te,
            y_pred=y_pred,
            n_samples=int(X.shape[0]),
        )

    raise ValueError("task must be 'classification' or 'regression'")


def main():
    rng = np.random.default_rng(0)
    n_demo = 40
    test_size = 0.25
    random_state = 42
    clf_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []

    iris = load_iris()
    wine_ds = load_wine()
    cancer_ds = load_breast_cancer()
    digits_ds = load_digits()
    Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=random_state)
    Xc, yc = make_circles(
        n_samples=300, noise=0.05, factor=0.5, random_state=random_state
    )
    Xb, yb = make_blobs(
        n_samples=300, centers=3, n_features=4, random_state=random_state
    )
    classification_runs: list[tuple[str, str, np.ndarray, np.ndarray, int, int]] = [
        ("Iris", "iris_species", iris.data, iris.target, n_demo, 35),
        ("Wine", "wine", wine_ds.data, wine_ds.target, min(60, len(wine_ds.target)), 35),
        (
            "Breast cancer",
            "breast_cancer",
            cancer_ds.data,
            cancer_ds.target,
            min(60, len(cancer_ds.target)),
            30,
        ),
        (
            "Digits",
            "digits",
            digits_ds.data,
            digits_ds.target,
            min(120, len(digits_ds.target)),
            25,
        ),
        ("Moons", "moons", Xm, ym, 80, 35),
        ("Circles", "circles", Xc, yc, 80, 35),
        ("Blobs", "blobs", Xb, yb, 80, 35),
    ]

    for title, csv_name, X, y, n_sub, maxiter_clf in classification_runs:
        print(f"\n=== {title} klasifikasi (subset) ===")
        Xs, ys = stratified_train_subset(X, y, n_sub, random_state)
        res_clf = adex_qml_train_eval(
            Xs,
            ys,
            task="classification",
            maxiter=maxiter_clf,
            test_size=test_size,
            random_state=random_state,
        )
        clf_rows.append(
            result_to_evaluation_row(
                res_clf,
                csv_name,
                maxiter_clf,
                test_size,
                random_state,
            )
        )

    print("\n=== Iris regresi: 3 fitur -> lebar kelopak ===")
    Xr = iris.data[:, :3]
    yr = iris.data[:, 3]
    idx_r = rng.choice(len(Xr), size=min(n_demo, len(Xr)), replace=False)
    maxiter_iris_reg = 35
    res_iris_reg = adex_qml_train_eval(
        Xr[idx_r],
        yr[idx_r],
        task="regression",
        maxiter=maxiter_iris_reg,
        test_size=test_size,
        random_state=random_state,
    )
    reg_rows.append(
        result_to_evaluation_row(
            res_iris_reg,
            "iris_petal_width",
            maxiter_iris_reg,
            test_size,
            random_state,
        )
    )

    print("\n=== Diabetes (regresi, subset) ===")
    dia = load_diabetes()
    n_d = min(35, len(dia.data))
    maxiter_dia = 30
    res_dia = adex_qml_train_eval(
        dia.data[:n_d],
        dia.target[:n_d],
        task="regression",
        maxiter=maxiter_dia,
        test_size=test_size,
        random_state=random_state,
    )
    reg_rows.append(
        result_to_evaluation_row(
            res_dia,
            "diabetes_progression",
            maxiter_dia,
            test_size,
            random_state,
        )
    )

    print("\n=== California housing (regresi, subset; unduh pertama kali) ===")
    cal = fetch_california_housing()
    n_c = min(30, len(cal.data))
    maxiter_cal = 25
    res_cal = adex_qml_train_eval(
        cal.data[:n_c],
        cal.target[:n_c],
        task="regression",
        maxiter=maxiter_cal,
        test_size=test_size,
        random_state=random_state,
    )
    reg_rows.append(
        result_to_evaluation_row(
            res_cal,
            "california_housing_median_house_value",
            maxiter_cal,
            test_size,
            random_state,
        )
    )

    out_dir = Path(__file__).resolve().parent
    clf_path, reg_path = write_evaluation_csvs(clf_rows, reg_rows, out_dir)
    print("\n=== Ekspor evaluasi (CSV) ===")
    print(f"  Klasifikasi: {clf_path}")
    print(f"  Regresi:     {reg_path}")


if __name__ == "__main__":
    main()
