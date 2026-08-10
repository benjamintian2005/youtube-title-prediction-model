import json
import os
import subprocess
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config
import thumbnails
from features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_preprocessor,
    engineer_title_features,
)


def _git_commit_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_benchmark(path=config.BENCHMARK_PATH):
    if not os.path.exists(path):
        raise SystemExit(
            f"No benchmark found at {path}.\n"
            f"Run 'python scrape.py' (if needed) then 'python freeze_benchmark.py' first."
        )
    return pd.read_csv(
        path,
        dtype={"video_id": str, "upload_date": str, "scraped_at": str, "split": str},
    )


def _add_channel_history_feature(df):
    # A channel's own past performance (independent of raw subscriber count -
    # e.g. upload cadence, audience engagement) turned out to be a plausible
    # missing signal, so this looks up each row's channel's average log(views/
    # day) among *other* videos from that channel. Split-aware and leak-safe:
    # stats are computed from split == 'train' rows only (so val/test targets
    # never leak into a feature), and a train row excludes its own log_vpd
    # from its own channel average (leave-one-out) so it can't trivially see
    # its own target. Channels with no other train-split video (the common
    # case - most channels appear once in this dataset) get NaN, imputed
    # downstream like duration/channel_follower_count.
    train_mask = df["split"] == "train"
    train_log_vpd = df.loc[train_mask, "log_vpd"]
    train_channel_id = df.loc[train_mask, "channel_id"]
    sums = train_log_vpd.groupby(train_channel_id).sum()
    counts = train_log_vpd.groupby(train_channel_id).count()

    def _hist_for_row(row):
        cid = row["channel_id"]
        if pd.isna(cid) or cid not in counts.index:
            return np.nan
        n = counts[cid]
        if row["split"] == "train":
            if n <= 1:
                return np.nan
            return (sums[cid] - row["log_vpd"]) / (n - 1)
        return sums[cid] / n

    return df.apply(_hist_for_row, axis=1)


def build_features(raw_df):
    # Filtering/feature logic is intentionally applied *after* loading the
    # frozen benchmark rather than baked into it, so changing MIN_DAYS_OLD or
    # engineer_title_features doesn't require re-freezing - the train/val/test
    # partition (by video_id) stays valid regardless.
    df = raw_df.copy()
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = None  # benchmark snapshot predates this feature
    if "thumbnail_url" not in df.columns:
        df["thumbnail_url"] = None  # benchmark snapshot predates this feature
    if "channel_id" not in df.columns:
        df["channel_id"] = None  # benchmark snapshot predates this feature
    df["upload_date"] = pd.to_datetime(df["upload_date"], format="%Y%m%d")
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], format="%Y%m%d")
    df["days_old"] = (df["scraped_at"] - df["upload_date"]).dt.days.clip(lower=1)

    df = df[df["days_old"] >= config.MIN_DAYS_OLD].reset_index(drop=True)

    df["views_per_day"] = df["view_count"] / df["days_old"]
    df["log_days_old"] = np.log1p(df["days_old"])
    df["log_vpd"] = np.log1p(df["views_per_day"])
    df["channel_hist_log_vpd"] = _add_channel_history_feature(df)

    title_feats = pd.DataFrame(df["title"].apply(engineer_title_features).tolist())
    df = pd.concat([df.reset_index(drop=True), title_feats], axis=1)

    # Downloads are cached under data/thumbnails/ (gitignored like data/videos.csv) -
    # only the first run against a given benchmark actually pays the network cost.
    thumb_feats = thumbnails.compute_features_for_df(df)
    df = pd.concat([df, thumb_feats], axis=1)

    feature_cols = ["title"] + NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    X = df[feature_cols]
    y = np.log1p(df["views_per_day"].to_numpy())
    return X, y, df["split"]


def build_candidates():
    return {
        "dummy": (Pipeline([("model", DummyRegressor(strategy="median"))]), None),
        "ridge": (
            Pipeline([("preprocess", build_preprocessor()), ("model", Ridge())]),
            config.RIDGE_PARAM_GRID,
        ),
        "hgb": (
            Pipeline([("preprocess", build_preprocessor(sparse=False)), ("model", HistGradientBoostingRegressor(random_state=42))]),
            config.HGB_PARAM_GRID,
        ),
    }


def evaluate_candidate(name, pipeline, param_grid, X_train, y_train, X_val, y_val):
    if param_grid:
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline.fit(X_train, y_train)
        best, best_params = pipeline, {}

    y_pred_log = best.predict(X_val)
    log_mse = mean_squared_error(y_val, y_pred_log)
    log_r2 = r2_score(y_val, y_pred_log)

    y_val_real = np.expm1(y_val)
    y_pred_real = np.expm1(y_pred_log)
    real_mae = mean_absolute_error(y_val_real, y_pred_real)
    mdape = float(np.median(np.abs((y_val_real - y_pred_real) / np.maximum(y_val_real, 1.0))) * 100)

    # column names kept as n_train/n_test for metrics.csv schema continuity;
    # n_test here is actually n_val - see build_features/main for the
    # train/val/test split (test is held out, see check_test.py).
    metrics = {
        "model_name": name,
        "best_params": json.dumps(best_params),
        "n_train": X_train.shape[0],
        "n_test": X_val.shape[0],
        "log_mse": log_mse,
        "log_r2": log_r2,
        "real_mae": real_mae,
        "mdape": mdape,
    }
    return best, metrics


def log_metrics(rows, path=config.METRICS_PATH):
    commit = _git_commit_hash()
    timestamp = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        row["timestamp"] = timestamp
        row["git_commit"] = commit
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    raw_df = load_benchmark()
    X, y, split = build_features(raw_df)
    print(f"{len(X)} rows remain after filtering days_old < {config.MIN_DAYS_OLD}")

    train_mask = (split == "train").to_numpy()
    val_mask = (split == "val").to_numpy()
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"train: {len(X_train)}  val: {len(X_val)}  (test set held out - see check_test.py)")

    results = []
    fitted = {}
    for name, (pipeline, param_grid) in build_candidates().items():
        best, metrics = evaluate_candidate(name, pipeline, param_grid, X_train, y_train, X_val, y_val)
        fitted[name] = best
        results.append(metrics)
        print(f"{name:>6}: log_mse={metrics['log_mse']:.4f}  log_r2={metrics['log_r2']:.4f}  "
              f"mae={metrics['real_mae']:.1f}  mdape={metrics['mdape']:.1f}%  params={metrics['best_params']}")

    best_name = min(
        (r for r in results if r["model_name"] != "dummy"),
        key=lambda r: r["log_mse"],
    )["model_name"]
    best_model = fitted[best_name]
    print(f"Best candidate this run: {best_name}")

    log_metrics(results)
    joblib.dump(best_model, config.CANDIDATE_PATH)
    print(f"Saved candidate to {config.CANDIDATE_PATH}")
    print(f"Compare its val log_mse against {config.CHAMPION_PATH} and promote manually if it wins (see program.md).")


if __name__ == "__main__":
    main()
