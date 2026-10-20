#!/usr/bin/env python3
"""Leakage-controlled EC521 FFN regression experiments and saved-model inference.

Run from the repository: .venv/bin/python ec521_submission/regression.py
The corrected original baseline preserves the 32-64-32-1 architecture, Adam
learning rate .001, batch size 32 and 50 epochs. Preprocessing is fit on each
training partition only. The common split manifest quarantines historical test
rows and keeps duplicate feature vectors in the same partition.

All reported errors and early-stopping decisions use original rspcexp23 units.
See protocol.json and selection.json for the fixed experiment and selection
rules. CV scores guide selection; their uncertainty is descriptive, not a
nested-CV estimate or significance test. The historical holdout is not pristine.
"""

from __future__ import annotations
from common import FEATURES as FEATURES32, final_partitions, load_data, make_partitions
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import torch
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(
    Path(__file__).parent / ".mplconfig"))
matplotlib.use("Agg")


VARIANTS = [
    {"name": "original", "dropout": 0.0, "early_stopping": False,
        "weight_decay": 0.0, "loss": "mse", "target": "standard"},
    {"name": "early_stopping", "dropout": 0.0, "early_stopping": True,
        "weight_decay": 0.0, "loss": "mse", "target": "standard"},
    {"name": "dropout", "dropout": 0.3, "early_stopping": False,
        "weight_decay": 0.0, "loss": "mse", "target": "standard"},
    {"name": "combined", "dropout": 0.3, "early_stopping": True,
        "weight_decay": 1e-4, "loss": "mse", "target": "standard"},
    {"name": "huber", "dropout": 0.0, "early_stopping": True,
        "weight_decay": 0.0, "loss": "smooth_l1_beta_1", "target": "standard"},
    {"name": "log_target", "dropout": 0.0, "early_stopping": True,
        "weight_decay": 0.0, "loss": "mse", "target": "log1p_standard"},
]
METRICS = ["rmse", "mae", "r2"]


def write_json(path, value):
    """Atomic JSON writes avoid incomplete records on interrupted runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2,
                    allow_nan=False), encoding="utf-8")
    temp.replace(path)


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not np.isfinite(pred).all():
        raise ValueError(
            "Nonfinite prediction: refusing to report invalid metrics.")
    residual = pred - y
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else 0.0,
        "mean_residual": float(np.mean(residual)),
        "p95_absolute_error": float(np.quantile(np.abs(residual), .95)),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "negative_prediction_fraction": float(np.mean(pred < 0)),
    }


class FFN(nn.Module):
    """Original two-hidden-layer network; dropout is active during training only."""

    def __init__(self, n_features=32, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(
                dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def inverse_target(pred_scaled, target_scaler, use_log):
    result = target_scaler.inverse_transform(np.asarray(
        pred_scaled).reshape(-1, 1)).ravel().astype(float)
    if use_log:
        result = np.expm1(result)
    # Intentionally no clipping/winsorization or log-bias correction.
    return result


def predict_tensor(model, x, target_scaler, use_log):
    model.eval()
    with torch.no_grad():
        scaled = np.concatenate([model(part).numpy().ravel()
                                for part in x.split(4096)])
    return inverse_target(scaled, target_scaler, use_log)


def fit_model(df, fit_ids, stop_ids, variant, seed, epochs):
    """Neither evaluation rows nor their labels are accepted by this function."""
    start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    fit_ids, stop_ids = np.asarray(fit_ids, int), np.asarray(stop_ids, int)
    assert not np.intersect1d(fit_ids, stop_ids).size
    features = list(FEATURES32)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    x_fit = torch.tensor(preprocessor.fit_transform(
        df.loc[fit_ids, features]), dtype=torch.float32)
    x_stop = torch.tensor(preprocessor.transform(
        df.loc[stop_ids, features]), dtype=torch.float32)
    y_fit = df.loc[fit_ids, "rspcexp23"].to_numpy(float)
    y_stop = df.loc[stop_ids, "rspcexp23"].to_numpy(float)
    use_log = variant["target"] == "log1p_standard"
    if use_log and np.any(y_fit < 0):
        raise ValueError("log1p target requires nonnegative consumption.")
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(
        (np.log1p(y_fit) if use_log else y_fit).reshape(-1, 1))
    dataset = torch.utils.data.TensorDataset(
        x_fit, torch.tensor(y_scaled, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = FFN(len(features), variant["dropout"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=.001, weight_decay=variant["weight_decay"])
    criterion = nn.MSELoss(
    ) if variant["loss"] == "mse" else nn.SmoothL1Loss(beta=1.0)
    best_rmse, best_epoch, stale, optimizer_steps = float("inf"), 0, 0, 0
    best_state, history = None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite {variant['name']} loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.item()) * len(xb)
        stop_pred = predict_tensor(model, x_stop, target_scaler, use_log)
        fit_pred = predict_tensor(model, x_fit, target_scaler, use_log)
        stop_rmse = float(np.sqrt(mean_squared_error(y_stop, stop_pred)))
        history.append({"epoch": epoch, "train_objective": total_loss / len(dataset),
                        "train_rmse": float(np.sqrt(mean_squared_error(y_fit, fit_pred))),
                        "stop_rmse": stop_rmse, "optimizer_steps": optimizer_steps})
        if stop_rmse < best_rmse - 1e-6:
            best_rmse, best_epoch, stale = stop_rmse, epoch, 0
            if variant["early_stopping"]:
                best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if variant["early_stopping"] and stale >= 5:
            break
    if variant["early_stopping"]:
        assert best_state is not None
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        # Baseline and dropout retain last-epoch weights regardless of stop scores.
        selected_epoch = epoch
    model.eval()
    metadata = {
        **variant, "seed": seed, "max_epochs": epochs, "epochs_run": epoch,
        "selected_epoch": selected_epoch, "optimizer_steps": optimizer_steps,
        "steps_at_selected_epoch": int(selected_epoch * math.ceil(len(dataset) / 32)),
        "n_fit": len(fit_ids), "n_stop": len(stop_ids),
        "training_seconds": time.perf_counter() - start,
        "learning_rate": .001, "batch_size": 32, "patience": 5,
        "selected_stop_rmse": history[selected_epoch - 1]["stop_rmse"],
        "features": features, "architecture": [len(features), 64, 32, 1],
        "device": "cpu", "torch_threads": 1, "target_units": "original_rspcexp23_units",
    }
    return model, preprocessor, target_scaler, metadata, history


def predict_rows(df, ids, fitted):
    model, preprocessor, target_scaler, metadata, _ = fitted
    x = torch.tensor(preprocessor.transform(
        df.loc[ids, metadata["features"]]), dtype=torch.float32)
    return predict_tensor(model, x, target_scaler, metadata["target"] == "log1p_standard")


def save_model(directory, fitted):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model, preprocessor, target_scaler, metadata, _ = fitted
    torch.save(model.state_dict(), directory / "weights.pt")
    joblib.dump({"features": preprocessor, "target": target_scaler},
                directory / "preprocessing.joblib")
    write_json(directory / "metadata.json", metadata)


def load_model(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    processors = joblib.load(directory / "preprocessing.joblib")
    model = FFN(len(metadata["features"]), metadata["dropout"])
    model.load_state_dict(torch.load(
        directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, processors["features"], processors["target"], metadata, []


def prediction_table(df, ids, pred, name, fold, seed, quintiles, tail_cut):
    ids = np.asarray(ids, int)
    data = pd.DataFrame({"row_index": ids, "variant": name, "fold": fold, "seed": seed,
                         "actual": df.loc[ids, "rspcexp23"].to_numpy(float), "predicted": pred})
    data["residual"] = data["predicted"] - data["actual"]
    data["absolute_error"] = np.abs(data["residual"])
    data["rural"] = df.loc[ids, "rural"].to_numpy()
    region = df.loc[ids, [f"reg_{r}" for r in range(1, 6)]]
    valid_region = region.eq(1).sum(axis=1).eq(1) & region.notna().all(axis=1)
    data["region"] = np.where(
        valid_region, region.fillna(-1).to_numpy().argmax(axis=1) + 1, 0)
    data["expenditure_quintile"] = np.searchsorted(
        quintiles, data["actual"], side="right") + 1
    data["upper_5pct_expenditure"] = data["actual"] >= tail_cut
    return data


def subgroup_table(predictions):
    rows = []
    for (variant, group_name), frame in predictions.groupby(["variant", "partition"], sort=False):
        for field in ["rural", "region", "expenditure_quintile", "upper_5pct_expenditure"]:
            for value, subset in frame.groupby(field, dropna=False):
                rows.append({"variant": variant, "partition": group_name, "group": field,
                             "value": str(value), "n_unique_rows": int(subset.row_index.nunique()),
                             **metrics(subset.actual, subset.predicted)})
    return pd.DataFrame(rows)


def select_model(cv_metrics, subgroups, folds):
    """Selection uses only development CV predictions, never holdout data."""
    fold_scores = cv_metrics.groupby(["variant", "fold"])[
        "rmse"].mean().unstack("variant")
    baseline = fold_scores["original"]
    baseline_sub = subgroups[(subgroups.variant == "original")
                             & subgroups.group.isin(["rural", "region"])]
    decisions = []
    for variant in [v["name"] for v in VARIANTS if v["name"] != "original"]:
        gains = baseline - fold_scores[variant]
        mean_gain = float(gains.mean())
        gain_se = float(gains.std(ddof=1) / np.sqrt(len(gains))
                        ) if len(gains) > 1 else 0.0
        fraction_gain = mean_gain / float(baseline.mean())
        bad_groups = []
        for base in baseline_sub.itertuples():
            candidate = subgroups[(subgroups.variant == variant) & (
                subgroups.group == base.group) & (subgroups.value == base.value)]
            if base.n_unique_rows >= 100 and len(candidate) == 1:
                mae_ratio = float(candidate.iloc[0].mae / base.mae)
                if mae_ratio > 1.10:
                    bad_groups.append(
                        {"group": base.group, "value": base.value, "mae_ratio": mae_ratio})
        improved_folds = int((gains > 0).sum())
        required_folds = math.ceil(.8 * folds)
        eligible = fraction_gain >= .01 and improved_folds >= required_folds and mean_gain > gain_se and not bad_groups
        decisions.append({"variant": variant, "mean_cv_rmse": float(fold_scores[variant].mean()),
                          "paired_rmse_gain": mean_gain, "relative_rmse_gain": fraction_gain,
                          "gain_standard_error": gain_se, "folds_improved": improved_folds,
                          "required_folds_improved": required_folds, "subgroup_mae_violations": bad_groups,
                          "eligible": bool(eligible)})
    eligible = [r for r in decisions if r["eligible"]]
    chosen = min(eligible, key=lambda r: r["mean_cv_rmse"])[
        "variant"] if eligible else "original"
    return {"selected_variant": chosen, "created_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_mean_cv_rmse": float(baseline.mean()), "candidates": decisions,
            "rule": "At least 1% mean RMSE reduction; positive gains in at least 80% of folds; mean paired gain greater than its standard error; no >10% rural/region subgroup MAE deterioration for >=100 distinct rows.",
            "caution": "Paired fold SE is a descriptive stability heuristic, not a significance test; folds overlap in training and CV is also used for model selection.",
            "holdout_used_for_selection": False}


def plot_results(cv, predictions, selected, histories, directory):
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = [v["name"] for v in VARIANTS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, METRICS):
        per_fold = cv.groupby(["variant", "fold"])[metric].mean()
        means = [per_fold.loc[name].mean() for name in names]
        deviations = [per_fold.loc[name].std(ddof=1) for name in names]
        ax.bar(range(len(names)), means, yerr=deviations, capsize=3,
               color=["#008B8B" if n == selected else "#92ADCA" for n in names])
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_ylabel(metric.upper())
    fig.suptitle(
        "Regression development CV: mean ± SD across folds (seeds averaged)")
    fig.tight_layout()
    fig.savefig(directory / "cv_comparison.png", dpi=160)
    plt.close(fig)
    selected_predictions = predictions[(predictions.variant == selected) & (
        predictions.partition == "historical_holdout")]
    actual, pred = selected_predictions.actual.to_numpy(
    ), selected_predictions.predicted.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(actual, pred, alpha=.18, s=8,
                    color="#237C8A", rasterized=True)
    low, high = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    axes[0].plot([low, high], [low, high], "--", color="#B34040")
    axes[0].set(xlabel="Observed expenditure", ylabel="Predicted expenditure",
                title="Full-range observed vs predicted")
    axes[1].scatter(pred, pred - actual, alpha=.18, s=8,
                    color="#237C8A", rasterized=True)
    axes[1].axhline(0, color="#B34040", linestyle="--")
    axes[1].set(xlabel="Predicted expenditure", ylabel="Prediction − observation",
                title="Residuals (no excluded extremes)")
    fig.suptitle(
        f"{selected}: historical holdout diagnostics (original units)")
    fig.tight_layout()
    fig.savefig(directory / "holdout_residuals.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, history in histories.items():
        frame = pd.DataFrame(history)
        ax.plot(frame.epoch, frame.train_rmse, label=f"{name}: fit")
        ax.plot(frame.epoch, frame.stop_rmse,
                linestyle="--", label=f"{name}: stopping")
    ax.set(xlabel="Epoch", ylabel="RMSE (original units)",
           title="Final development training; dashed lines use stopping data")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "final_learning_curves.png", dpi=160)
    plt.close(fig)
    subgroup = subgroup_table(predictions)
    quintile = subgroup[(subgroup.partition == "historical_holdout") & (
        subgroup.group == "expenditure_quintile")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, frame in quintile.groupby("variant"):
        frame = frame.sort_values("value")
        ax.plot(frame.value, frame.mae, marker="o", label=name)
    ax.set(xlabel="Expenditure quintile (cut points from development labels)", ylabel="MAE (original units)",
           title="Historical holdout: errors across the consumption distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "holdout_quintile_errors.png", dpi=160)
    plt.close(fig)


def inference(args):
    if args.model_dir is None:
        raise SystemExit("--model-dir is required with --predict")
    fitted = load_model(args.model_dir)
    frame = pd.read_csv(args.predict, sep=None, engine="python")
    frame.columns = frame.columns.str.strip()
    missing = sorted(set(fitted[3]["features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    frame = frame[fitted[3]["features"]].copy()
    for name in frame:
        values = frame[name].map(
            lambda v: v.strip() if isinstance(v, str) else v)
        values = values.map(lambda v: {"yes": 1, "no": 0}.get(v.lower(), v)
                            if isinstance(v, str) else v)
        numeric = pd.to_numeric(values, errors="coerce")
        if (values.notna() & numeric.isna()).any():
            raise ValueError(f"Unrecognized nonnumeric values in {name}")
        frame[name] = numeric
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite input features are not accepted")
    torch.set_num_threads(1)
    predicted = predict_rows(frame, frame.index.to_numpy(), fitted)
    output = Path(args.prediction_output or "regression_predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": frame.index, "predicted_rspcexp23": predicted}).to_csv(
        output, index=False)
    print(f"Saved {len(frame)} predictions to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent.parent / "SES2023.csv")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,42")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke", action="store_true",
                        help="Run real data for 2 epochs/2 folds/one seed in a separate smoke directory")
    parser.add_argument("--predict", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()
    if args.predict:
        inference(args)
        return
    if args.epochs < 1 or args.folds < 2:
        raise ValueError("Require epochs >=1 and folds >=2")
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique integers")
    if args.smoke:
        args.output_root = args.output_root / "smoke"
        args.epochs, args.folds, seeds = 2, 2, [0]
    root = args.output_root.resolve()
    result_dir, figure_dir, model_dir, log_dir = [
        root / name / "regression" for name in ["results", "figures", "models", "logs"]]
    for directory in [result_dir, figure_dir, model_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    partitions = make_partitions(
        df, root, n_splits=args.folds, split_seed=20261020)
    quintiles = np.quantile(
        df.loc[partitions["dev"], "rspcexp23"], [.2, .4, .6, .8])
    tail_cut = float(np.quantile(df.loc[partitions["dev"], "rspcexp23"], .95))
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "task": "regression", "smoke": args.smoke,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "n_rows": len(df),
        "n_development": len(partitions["dev"]), "n_historical_holdout": len(partitions["holdout"]),
        "features": list(FEATURES32), "folds": args.folds, "seeds": seeds, "split_seed": 20261020,
        "variants": VARIANTS, "maximum_epochs": args.epochs,
        "training_partition": "fit + calibration for every regression variant; stop and evaluation remain separate",
        "preprocessing": "Training-only median imputation and feature scaling; training-only target scaling. No target imputation.",
        "early_stopping": "Patience 5 on original-unit stopping RMSE; restore best weights; no stopping selection for fixed-epoch variants.",
        "selection_rule": "At least 1% paired mean fold RMSE reduction, >=80% positive fold gains, mean paired gain > its SE, no >10% rural/region MAE deterioration where n_unique>=100; otherwise original. Lowest RMSE eligible model wins.",
        "expenditure_quintile_cuts": quintiles.tolist(), "upper_5pct_cut": tail_cut,
        "uncertainty": "Summaries average seeds within folds before calculating SD/SE; paired fold heuristic is not a significance test.",
        "holdout_caution": "Historical test-row union is quarantined from development but not an unseen external test.",
        "compute_fairness": "Same fit rows, batches, maximum epochs and seeds; early stopping changes actual optimizer steps, reported per run.",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    write_json(result_dir / "protocol.json", protocol)
    rows, predictions = [], []
    for fold_number, split in enumerate(partitions["folds"], start=1):
        fit_ids = np.sort(np.concatenate([split["fit"], split["calibration"]]))
        stop_ids, eval_ids = split["stop"], split["evaluation"]
        assert len(set(fit_ids) & set(eval_ids)) == 0 and len(
            set(stop_ids) & set(eval_ids)) == 0
        for seed in seeds:
            for variant in VARIANTS:
                run_id = f"fold{fold_number}_seed{seed}_{variant['name']}"
                print(f"Training {run_id}", flush=True)
                fitted = fit_model(df, fit_ids, stop_ids,
                                   variant, seed, args.epochs)
                pred = predict_rows(df, eval_ids, fitted)
                row = {"variant": variant["name"], "fold": fold_number, "seed": seed,
                       **metrics(df.loc[eval_ids, "rspcexp23"], pred),
                       **{key: fitted[3][key] for key in ["epochs_run", "selected_epoch", "optimizer_steps", "steps_at_selected_epoch", "training_seconds", "n_fit", "n_stop"]}}
                rows.append(row)
                table = prediction_table(
                    df, eval_ids, pred, variant["name"], fold_number, seed, quintiles, tail_cut)
                table["partition"] = "development_cv"
                predictions.append(table)
                pd.DataFrame(fitted[4]).to_csv(
                    log_dir / f"{run_id}_history.csv", index=False)
                table.to_csv(
                    result_dir / f"{run_id}_predictions.csv", index=False)
                write_json(
                    log_dir / f"{run_id}.json", {"configuration": fitted[3], "evaluation": row})
                pd.DataFrame(rows).to_csv(
                    result_dir / "cv_runs.csv", index=False)
                print(
                    f"  RMSE={row['rmse']:.3f}, MAE={row['mae']:.3f}, R2={row['r2']:.5f}; epochs={row['epochs_run']}; {row['training_seconds']:.1f}s", flush=True)
    cv = pd.DataFrame(rows)
    per_fold = cv.groupby(["variant", "fold"], as_index=False)[METRICS].mean()
    per_fold.to_csv(result_dir / "cv_fold_seed_means.csv", index=False)
    summary = per_fold.groupby("variant")[METRICS].agg(["mean", "std", "sem"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(result_dir / "cv_summary.csv")
    cv_predictions = pd.concat(predictions, ignore_index=True)
    cv_predictions.to_csv(result_dir / "cv_predictions.csv", index=False)
    cv_subgroups = subgroup_table(cv_predictions)
    cv_subgroups.to_csv(
        result_dir / "cv_subgroup_diagnostics.csv", index=False)
    selection = select_model(cv, cv_subgroups, args.folds)
    # Persist the full selection decision before any final model/holdout prediction.
    write_json(result_dir / "selection.json", selection)
    selection_sha256 = hashlib.sha256(
        (result_dir / "selection.json").read_bytes()).hexdigest()
    selected = selection["selected_variant"]
    print(
        f"Selection locked before holdout evaluation: {selected}", flush=True)
    final_split = final_partitions(df, partitions["dev"], split_seed=20261020)
    final_fit = np.sort(np.concatenate(
        [final_split["fit"], final_split["calibration"]]))
    final_rows, final_tables, histories = [], [], {}
    reload_checks = []
    for name in dict.fromkeys(["original", selected]):
        variant = next(v for v in VARIANTS if v["name"] == name)
        fitted = fit_model(
            df, final_fit, final_split["stop"], variant, 0, args.epochs)
        save_model(model_dir / name, fitted)
        pred = predict_rows(df, partitions["holdout"], fitted)
        row = {"variant": name, "selected": name == selected, "seed": 0,
               **metrics(df.loc[partitions["holdout"], "rspcexp23"], pred),
               **{key: fitted[3][key] for key in ["epochs_run", "selected_epoch", "optimizer_steps", "training_seconds", "n_fit", "n_stop"]}}
        final_rows.append(row)
        table = prediction_table(
            df, partitions["holdout"], pred, name, "final", 0, quintiles, tail_cut)
        table["partition"] = "historical_holdout"
        final_tables.append(table)
        histories[name] = fitted[4]
        pd.DataFrame(fitted[4]).to_csv(
            log_dir / f"final_{name}_history.csv", index=False)
        write_json(log_dir / f"final_{name}.json",
                   {"configuration": fitted[3], "evaluation": row})
        reloaded = predict_rows(
            df, partitions["holdout"], load_model(model_dir / name))
        delta = float(np.max(np.abs(pred - reloaded)))
        assert delta == 0.0, f"Saved/reloaded predictions differ by {delta}"
        reload_checks.append({"variant": name, "n_predictions": len(
            pred), "maximum_absolute_reload_difference": delta})
    final_predictions = pd.concat(final_tables, ignore_index=True)
    final_predictions.to_csv(
        result_dir / "holdout_predictions.csv", index=False)
    pd.DataFrame(final_rows).to_csv(
        result_dir / "holdout_metrics.csv", index=False)
    subgroup_table(final_predictions).to_csv(
        result_dir / "holdout_subgroup_diagnostics.csv", index=False)
    all_predictions = pd.concat(
        [cv_predictions, final_predictions], ignore_index=True)
    plot_results(cv, all_predictions, selected, histories, figure_dir)
    write_json(result_dir / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "selected_variant": selected, "cv_runs": len(rows), "selection_sha256_before_holdout": selection_sha256,
                                                "selection_sha256_after_holdout": hashlib.sha256((result_dir / "selection.json").read_bytes()).hexdigest(),
                                                "reload_checks": reload_checks, "smoke": args.smoke})
    print(pd.DataFrame(final_rows).to_string(index=False), flush=True)
    print(f"Completed regression artifacts: {root}", flush=True)


if __name__ == "__main__":
    main()
