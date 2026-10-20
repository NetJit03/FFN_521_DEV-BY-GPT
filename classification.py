#!/usr/bin/env python3
"""Leakage-controlled FFN classification experiments and saved-model inference.

The original 32 -> 64 -> 32 -> 1 ReLU architecture, Adam learning rate,
batch size, and maximum epoch count are retained. The baseline is a corrected
re-evaluation, not a claim to reproduce the score under the earlier evaluation protocol.
Only fit rows train preprocessing/weights; stop rows select early stopping;
calibration rows choose thresholds; outer evaluation rows report CV metrics.
"""
from __future__ import annotations
from common import FEATURES, load_data, make_partitions, final_partitions
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from torch import nn
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(
    Path(__file__).parent / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

matplotlib.use("Agg")


CANDIDATES = {
    "baseline": dict(pos_weight=1.0, dropout=0.0, weight_decay=0.0, early_stopping=False),
    "weight_3": dict(pos_weight=3.0, dropout=0.0, weight_decay=0.0, early_stopping=False),
    "weight_10": dict(pos_weight=10.0, dropout=0.0, weight_decay=0.0, early_stopping=False),
    "weight_ratio": dict(pos_weight="fit_ratio", dropout=0.0, weight_decay=0.0, early_stopping=False),
    "ros_0.10": dict(pos_weight=1.0, dropout=0.0, weight_decay=0.0, early_stopping=False, sampling_ratio=0.10),
    "regularized": dict(pos_weight=1.0, dropout=0.20, weight_decay=1e-4, early_stopping=True),
}
METRICS = ["precision", "recall", "f1", "average_precision", "accuracy"]
SELECTION_RULE = (
    "Incumbent is baseline with independently calibrated threshold. Average seeds "
    "within each fold. A challenger qualifies only if its paired F1 gain is positive "
    "in >=4/5 folds (ceil(0.8*n_folds)) and its mean gain exceeds the standard error "
    "of paired fold gains. Choose the qualifying model with highest mean F1, with "
    "alphabetical tie-breaking. This is a stability heuristic, not a significance test."
)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


class FFN(nn.Module):
    def __init__(self, in_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(
                dropout), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def predict_scores(model: FFN, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        chunks = [torch.sigmoid(model(torch.from_numpy(batch))).numpy().ravel()
                  for batch in np.array_split(x, max(1, math.ceil(len(x) / 8192)))]
    result = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite prediction encountered")
    return result


def optimize_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Maximize calibration F1 over unique scores; ties favor highest threshold."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    best_indices = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    return float(thresholds[best_indices[-1]])


def evaluate(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        average_precision=average_precision_score(y, scores),
        accuracy=accuracy_score(y, pred), n=len(y), positives=int(y.sum()),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def train_model(xfit, yfit, xstop, ystop, config, seed, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = FFN(xfit.shape[1], config["dropout"])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config["resolved_pos_weight"]]))
    stop_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=config["weight_decay"])
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(
        xfit), torch.from_numpy(yfit.astype(np.float32)).reshape(-1, 1))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed), num_workers=0)
    xstop_t = torch.from_numpy(xstop)
    ystop_t = torch.from_numpy(ystop.astype(np.float32)).reshape(-1, 1)
    best_loss, best_state, best_epoch, stale, steps = np.inf, None, 0, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            steps += 1
        model.eval()
        with torch.no_grad():
            stop_loss = float(stop_criterion(model(xstop_t), ystop_t))
        history.append(dict(epoch=epoch, training_objective=running_loss / len(dataset),
                            stop_unweighted_bce=stop_loss, optimizer_steps=steps))
        if stop_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            if config["early_stopping"]:
                best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if config["early_stopping"] and stale >= 7:
            break
    if config["early_stopping"]:
        model.load_state_dict(best_state)
        retained_epoch = best_epoch
    else:
        retained_epoch = len(history)
    info = dict(epochs_executed=len(history), retained_epoch=retained_epoch,
                best_stop_epoch=best_epoch, optimizer_steps=steps,
                retained_optimizer_steps=retained_epoch * len(loader),
                training_rows_after_resampling=len(dataset), seconds=time.perf_counter() - started)
    return model, history, info


def fit_run(df, partitions, method, seed, epochs, paths, run_name):
    """Run one fold without exposing evaluation labels to any fitted choice."""
    config = dict(CANDIDATES[method])
    yfit = df.loc[partitions["fit"], "pov_exp"].to_numpy(dtype=int)
    config["resolved_pos_weight"] = (
        float((yfit == 0).sum() / (yfit == 1).sum())
        if config["pos_weight"] == "fit_ratio" else float(config["pos_weight"])
    )
    processor = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("scaler", StandardScaler())])
    processor.fit(df.loc[partitions["fit"], FEATURES])
    xs = {name: processor.transform(df.loc[idx, FEATURES]).astype(np.float32)
          for name, idx in partitions.items()}
    ys = {name: df.loc[idx, "pov_exp"].to_numpy(
        dtype=int) for name, idx in partitions.items()}
    for array in xs.values():
        if not np.isfinite(array).all():
            raise ValueError("Preprocessing produced non-finite features")
    xtrain, ytrain = xs["fit"], ys["fit"]
    sampler_indices = np.arange(len(ytrain))
    if "sampling_ratio" in config:
        sampler = RandomOverSampler(
            sampling_strategy=config["sampling_ratio"], random_state=seed)
        xtrain, ytrain = sampler.fit_resample(xtrain, ytrain)
        sampler_indices = sampler.sample_indices_
    model, history, training = train_model(
        xtrain, ytrain, xs["stop"], ys["stop"], config, seed, epochs)
    scores = {name: predict_scores(model, array) for name, array in xs.items()}
    threshold = optimize_threshold(ys["calibration"], scores["calibration"])
    info = dict(method=method, seed=seed, threshold=threshold, config=config, **training,
                training_rows_before_resampling=len(yfit), fit_positives=int(yfit.sum()),
                architecture=[len(FEATURES), 64, 32, 1], activation="ReLU",
                optimizer="Adam", learning_rate=0.001, batch_size=64, max_epochs=epochs,
                stop_patience=7, stop_min_delta=1e-6, device="cpu", torch_threads=1,
                threshold_selection="calibration F1; unique scores; highest threshold on ties",
                features=FEATURES, run_name=run_name,
                split_sizes={name: len(idx) for name, idx in partitions.items()})
    folder = paths["models"] / run_name
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "weights.pt")
    joblib.dump(processor, folder / "preprocessor.joblib")
    save_json(folder / "metadata.json", info)
    np.savez_compressed(folder / "row_indices.npz", **
                        partitions, resampled_fit_positions=sampler_indices)
    pd.DataFrame(history).to_csv(
        paths["logs"] / f"{run_name}_history.csv", index=False)
    save_json(paths["logs"] / f"{run_name}_config.json", info)
    np.savez_compressed(paths["results"] / f"{run_name}_predictions.npz",
                        **{f"{k}_row_index": v for k, v in partitions.items()},
                        **{f"{k}_y": v for k, v in ys.items()},
                        **{f"{k}_score": v for k, v in scores.items()})
    metrics = []
    for split in ("fit", "stop", "calibration", "evaluation"):
        if split not in partitions:
            continue
        for mode, cut in [("fixed_0.5", 0.5), ("calibrated", threshold)]:
            metrics.append(dict(method=method, seed=seed, split=split, threshold_mode=mode,
                                threshold=cut, **evaluate(ys[split], scores[split], cut),
                                **training, training_rows_before_resampling=len(yfit)))
    return info, metrics, scores, ys


def select_candidate(cv: pd.DataFrame):
    evaluation = cv.query(
        "split == 'evaluation' and threshold_mode == 'calibrated'")
    folded = evaluation.groupby(["method", "fold"])[
        "f1"].mean().unstack("method")
    rows = []
    for method in CANDIDATES:
        gains = folded[method] - folded["baseline"]
        gain_se = float(gains.std(ddof=1) / np.sqrt(len(gains)))
        qualifying = method != "baseline" and (gains > 0).sum(
        ) >= math.ceil(.8 * len(gains)) and gains.mean() > gain_se
        rows.append(dict(method=method, mean_f1=float(folded[method].mean()),
                         paired_mean_gain=float(gains.mean()), paired_gain_se=gain_se,
                         positive_fold_gains=int((gains > 0).sum()), eligible=bool(qualifying)))
    comparison = pd.DataFrame(rows)
    eligible = comparison[comparison.eligible].sort_values(
        ["mean_f1", "method"], ascending=[False, True])
    chosen = "baseline" if eligible.empty else str(eligible.iloc[0].method)
    return chosen, comparison


def write_cv_tables(cv: pd.DataFrame, paths):
    cv.to_csv(paths["results"] / "cv_metrics.csv", index=False)
    evaluation = cv.query("split == 'evaluation'")
    # Seeds share fold rows, so the dispersion reported below is across five
    # fold means, rather than treating ten correlated runs as independent.
    fold_means = evaluation.groupby(
        ["method", "threshold_mode", "fold"], as_index=False)[METRICS].mean()
    fold_means.to_csv(paths["results"] / "cv_fold_means.csv", index=False)
    summary = fold_means.groupby(["method", "threshold_mode"])[
        METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(pair) for pair in summary.columns]
    summary.reset_index().to_csv(
        paths["results"] / "cv_summary.csv", index=False)
    cost = evaluation.query("threshold_mode == 'calibrated'").groupby("method")[[
        "epochs_executed", "optimizer_steps", "seconds", "training_rows_after_resampling"
    ]].agg(["mean", "min", "max"])
    cost.columns = ["_".join(pair) for pair in cost.columns]
    cost.to_csv(paths["results"] / "training_cost.csv")


def make_plots(cv, holdout, final_scores, final_ys, final_infos, paths):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    ev = cv.query("split == 'evaluation'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mode, marker in [("fixed_0.5", "o"), ("calibrated", "s")]:
        group = ev.query(
            "threshold_mode == @mode").groupby(["method", "fold"])[METRICS].mean().reset_index()
        summary = group.groupby("method")[METRICS].agg(
            ["mean", "std"]).reindex(CANDIDATES)
        for ax, metric in zip(axes, ["f1", "average_precision"]):
            ax.errorbar(np.arange(len(CANDIDATES)), summary[(metric, "mean")],
                        yerr=summary[(metric, "std")], fmt=marker, capsize=3, label=mode)
            ax.set_xticks(np.arange(len(CANDIDATES)),
                          CANDIDATES, rotation=35, ha="right")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(alpha=.2)
    axes[0].legend()
    axes[0].set_title("Outer CV F1 (mean ± fold SD)")
    axes[1].set_title("Ranking quality (threshold-independent)")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "cv_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, scores in final_scores.items():
        precision, recall, _ = precision_recall_curve(final_ys, scores)
        ap = average_precision_score(final_ys, scores)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.axhline(final_ys.mean(), color="gray", ls="--", label="Prevalence")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Historical holdout: precision–recall (diagnostic)", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "holdout_precision_recall.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(holdout), figsize=(5 * len(holdout), 4.5))
    for ax, (_, row) in zip(np.atleast_1d(axes), holdout.iterrows()):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center",
                    color="white" if value > cm.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-poor", "Poor"],
               yticklabels=["Non-poor", "Poor"], xlabel="Predicted", ylabel="Actual",
               title=f"{row.label}\nF1={row.f1:.3f}; P={row.precision:.3f}; R={row.recall:.3f}")
    fig.suptitle("Historical holdout confusion matrices (not a pristine test)")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "holdout_confusion_matrices.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in final_infos:
        h = pd.read_csv(paths["logs"] / f"final_{method}_history.csv")
        ax.plot(h.epoch, h.stop_unweighted_bce, label=f"{method}: stop BCE")
        ax.plot(h.epoch, h.training_objective, ls="--",
                label=f"{method}: training objective")
    ax.set(xlabel="Epoch", ylabel="Loss",
           title="Final fits: training and stopping curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "final_learning_curves.png")
    plt.close(fig)


def inference(args):
    folder = Path(args.model_dir)
    metadata = json.loads((folder / "metadata.json").read_text())
    frame = pd.read_csv(args.predict, sep=None, engine="python")
    frame.columns = frame.columns.str.strip()
    missing = set(metadata["features"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")
    x = frame[metadata["features"]].copy()
    for name in x:
        values = x[name].map(lambda v: v.strip() if isinstance(v, str) else v)
        values = values.map(lambda v: {"yes": 1, "no": 0}.get(v.lower(), v)
                            if isinstance(v, str) else v)
        numeric = pd.to_numeric(values, errors="coerce")
        if (values.notna() & numeric.isna()).any():
            raise ValueError(f"Unrecognized nonnumeric values in {name}")
        x[name] = numeric
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Infinite input features are not accepted")
    processor = joblib.load(folder / "preprocessor.joblib")
    ready = processor.transform(x).astype(np.float32)
    model = FFN(len(metadata["features"]), metadata["config"]["dropout"])
    model.load_state_dict(torch.load(
        folder / "weights.pt", map_location="cpu", weights_only=True))
    scores = predict_scores(model, ready)
    out = Path(args.prediction_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": frame.index, "poverty_score": scores,
                  "predicted_pov_exp": (scores >= metadata["threshold"]).astype(int),
                  "threshold": metadata["threshold"]}).to_csv(out, index=False)
    print(f"Saved {len(frame)} predictions to {out}", flush=True)


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
                        help="Real-data 6,000-row, two-epoch integration run in smoke/ only")
    parser.add_argument("--predict", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prediction-output", type=Path,
                        default=Path("classification_predictions.csv"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.predict:
        if not args.model_dir:
            parser.error("--predict requires --model-dir")
        inference(args)
        return
    if args.folds < 2 or args.epochs < 1:
        parser.error("Require at least two folds and one epoch")
    seeds = [int(value) for value in args.seeds.split(",")]
    root = args.output_root / "smoke" if args.smoke else args.output_root
    paths = {name: root / name /
             "classification" for name in ("results", "figures", "models", "logs")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    if args.smoke:
        if len(df) > 6000:
            idx, _ = train_test_split(df.index.to_numpy(
            ), train_size=6000, stratify=df.pov_exp, random_state=20261020)
            save_json(paths["results"] / "smoke_original_rows.json",
                      {"original_row_indices": np.sort(idx)})
            df = df.loc[np.sort(idx)].reset_index(drop=True)
        args.epochs, seeds = 2, [0]
    partitions = make_partitions(
        df, paths["results"] / "splits", n_splits=args.folds, split_seed=20261020)
    protocol = dict(data=str(args.data.resolve()), smoke=args.smoke, rows=len(df),
                    seeds=seeds, folds=args.folds, max_epochs=args.epochs,
                    candidates=CANDIDATES, selection_rule=SELECTION_RULE,
                    final_seed=0, holdout_role="historically exposed diagnostic; not a pristine test",
                    smotenc="Not run: household-share composition and one-hot constraints require careful joint synthesis; minority is small. ROS preserves observed vectors.",
                    baseline="Original architecture/optimizer; corrected group-disjoint preprocessing and evaluation protocol",
                    starting_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    save_json(paths["results"] / "protocol.json", protocol)
    all_metrics = []
    for fold_no, fold in enumerate(partitions["folds"], 1):
        for seed in seeds:
            for method in CANDIDATES:
                run_name = f"fold{fold_no}_seed{seed}_{method}"
                print(f"START {run_name}", flush=True)
                info, metrics, _, _ = fit_run(
                    df, fold, method, seed, args.epochs, paths, run_name)
                all_metrics.extend([dict(fold=fold_no, **row)
                                   for row in metrics])
                write_cv_tables(pd.DataFrame(all_metrics), paths)
                ev = next(
                    r for r in metrics if r["split"] == "evaluation" and r["threshold_mode"] == "calibrated")
                print(
                    f"DONE {run_name}: F1={ev['f1']:.4f}, P={ev['precision']:.4f}, R={ev['recall']:.4f}, AP={ev['average_precision']:.4f}, steps={info['optimizer_steps']}, seconds={info['seconds']:.1f}", flush=True)
    cv = pd.DataFrame(all_metrics)
    chosen, comparison = select_candidate(cv)
    comparison.to_csv(paths["results"] /
                      "selection_comparison.csv", index=False)
    locked_selection = dict(method=chosen, final_seed=0, rule=SELECTION_RULE,
                            selection_time=time.strftime(
                                "%Y-%m-%dT%H:%M:%S%z"),
                            holdout_used_for_selection=False,
                            limitation="Outer CV estimates are used to select the method and are subject to selection optimism. Holdout rows were exposed by historical work.")
    save_json(paths["results"] / "locked_selection.json", locked_selection)
    print(
        f"LOCKED SELECTION: {chosen}; final seed=0. Starting diagnostic holdout evaluation.", flush=True)
    final_split = final_partitions(df, partitions["dev"])
    final_split["evaluation"] = partitions["holdout"]
    final_infos, final_scores, final_truth = {}, {}, None
    rows = []
    for method in dict.fromkeys(["baseline", chosen]):
        info, metrics, scores, ys = fit_run(
            df, final_split, method, 0, args.epochs, paths, f"final_{method}")
        final_infos[method] = info
        final_scores[method] = scores["evaluation"]
        final_truth = ys["evaluation"]
        for mode, label in ([("fixed_0.5", "Baseline at 0.5"), ("calibrated", "Baseline + threshold")] if method == "baseline" else [("calibrated", f"Selected: {method}")]):
            ev = next(r for r in metrics if r["split"] ==
                      "evaluation" and r["threshold_mode"] == mode)
            rows.append(
                dict(label=label, role="historical_holdout_diagnostic", **ev))
        pd.DataFrame(metrics).to_csv(
            paths["results"] / f"final_{method}_metrics.csv", index=False)
    selected_folder = paths["models"] / "selected"
    selected_folder.mkdir(parents=True, exist_ok=True)
    for name in ("weights.pt", "preprocessor.joblib", "metadata.json", "row_indices.npz"):
        shutil.copy2(paths["models"] /
                     f"final_{chosen}" / name, selected_folder / name)
    holdout = pd.DataFrame(rows)
    holdout.to_csv(paths["results"] /
                   "historical_holdout_metrics.csv", index=False)
    prediction_table = pd.DataFrame(
        {"row_index": final_split["evaluation"], "y_true": final_truth})
    for method, scores in final_scores.items():
        prediction_table[f"{method}_score"] = scores
        prediction_table[f"{method}_prediction_calibrated"] = (
            scores >= final_infos[method]["threshold"]).astype(int)
    prediction_table.to_csv(
        paths["results"] / "historical_holdout_predictions.csv", index=False)
    make_plots(cv, holdout, final_scores, final_truth, final_infos, paths)
    print(holdout[["label", *METRICS]].to_string(index=False), flush=True)
    save_json(paths["results"] / "completion.json", dict(complete=True, smoke=args.smoke, selected=chosen,
              cv_training_runs=len(
                  partitions["folds"]) * len(seeds) * len(CANDIDATES),
              output_root=str(root.resolve()), finished=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


if __name__ == "__main__":
    main()
