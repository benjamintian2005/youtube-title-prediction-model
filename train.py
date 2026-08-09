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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import config
import scrape
from features import NUMERIC_FEATURE_COLUMNS, build_preprocessor, engineer_title_features


def _git_commit_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_or_scrape():
    df = scrape.load_cached()
    if df.empty:
        print("No cached data found, scraping...")
        records = scrape.fetch_data(config.SEED_WORDS)
        df = scrape.save_and_merge(records)
    return df


def build_dataset(raw_df):
    df = raw_df.copy()
    df["upload_date"] = pd.to_datetime(df["upload_date"], format="%Y%m%d")
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], format="%Y%m%d")
    df["days_old"] = (df["scraped_at"] - df["upload_date"]).dt.days.clip(lower=1)

    df = df[df["days_old"] >= config.MIN_DAYS_OLD].reset_index(drop=True)

    df["views_per_day"] = df["view_count"] / df["days_old"]
    df["log_days_old"] = np.log1p(df["days_old"])

    title_feats = pd.DataFrame(df["title"].apply(engineer_title_features).tolist())
    df = pd.concat([df.reset_index(drop=True), title_feats], axis=1)

    feature_cols = ["title"] + NUMERIC_FEATURE_COLUMNS
    X = df[feature_cols]
    y = np.log1p(df["views_per_day"].to_numpy())
    return X, y


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


def evaluate_candidate(name, pipeline, param_grid, X_train, y_train, X_test, y_test):
    if param_grid:
        search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = search.best_params_
    else:
        pipeline.fit(X_train, y_train)
        best, best_params = pipeline, {}

    y_pred_log = best.predict(X_test)
    log_mse = mean_squared_error(y_test, y_pred_log)
    log_r2 = r2_score(y_test, y_pred_log)

    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)
    real_mae = mean_absolute_error(y_test_real, y_pred_real)
    mdape = float(np.median(np.abs((y_test_real - y_pred_real) / np.maximum(y_test_real, 1.0))) * 100)

    metrics = {
        "model_name": name,
        "best_params": json.dumps(best_params),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
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
    raw_df = load_or_scrape()
    print(f"Loaded {len(raw_df)} cached rows")

    X, y = build_dataset(raw_df)
    print(f"{len(X)} rows remain after filtering days_old < {config.MIN_DAYS_OLD}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    fitted = {}
    for name, (pipeline, param_grid) in build_candidates().items():
        best, metrics = evaluate_candidate(name, pipeline, param_grid, X_train, y_train, X_test, y_test)
        fitted[name] = best
        results.append(metrics)
        print(f"{name:>6}: log_mse={metrics['log_mse']:.4f}  log_r2={metrics['log_r2']:.4f}  "
              f"mae={metrics['real_mae']:.1f}  mdape={metrics['mdape']:.1f}%  params={metrics['best_params']}")

    best_name = min(
        (r for r in results if r["model_name"] != "dummy"),
        key=lambda r: r["log_mse"],
    )["model_name"]
    best_model = fitted[best_name]
    print(f"Best model: {best_name}")

    log_metrics(results)
    joblib.dump(best_model, config.MODEL_PATH)
    print(f"Saved {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
