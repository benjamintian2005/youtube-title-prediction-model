import sys

import joblib
import numpy as np
import pandas as pd

import config
from features import CATEGORICAL_FEATURE_COLUMNS, NUMERIC_FEATURE_COLUMNS, engineer_title_features
from thumbnails import THUMBNAIL_FEATURE_COLUMNS

SAMPLE_TITLES = [
    "How to train a neural network",
    "Top 10 programming languages in 2024",
    "Beginner's guide to machine learning",
    "Best practices for software development",
    "Advanced Python tutorials",
]


def load_model(path=config.MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run train.py first to train and save the model.")
        sys.exit(1)


PROJECTION_DAYS = [30, 365]


def build_input_row(title, days_old=None):
    feats = engineer_title_features(title)
    feats["title"] = title
    # duration/channel_follower_count are unknown at title-only inference
    # time; the pipeline's imputer fills them from training medians.
    # days_old is provided per-call (see predict_views_per_day) since the
    # model learned a strong age/velocity relationship (see features.py)
    # and a single imputed age would silently bias every projection.
    feats["log_days_old"] = np.log1p(days_old) if days_old is not None else np.nan
    feats["duration"] = np.nan
    feats["channel_follower_count"] = np.nan
    # category is knowable pre-publish in principle, but predict.py's CLI is
    # title-only for now - the pipeline's imputer buckets it as "unknown".
    feats["category"] = np.nan
    # thumbnail features need an actual image; unlike category, there's no
    # thumbnail to point to for a title that doesn't exist as a video yet.
    for col in THUMBNAIL_FEATURE_COLUMNS:
        feats[col] = np.nan
    cols = ["title"] + NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    return pd.DataFrame([{col: feats[col] for col in cols}])


def predict_views_per_day(title, model, days_old=None):
    row = build_input_row(title, days_old=days_old)
    return float(np.expm1(model.predict(row)[0]))


def main():
    model = load_model()
    titles = sys.argv[1:] if len(sys.argv) > 1 else SAMPLE_TITLES

    for title in titles:
        vpd_day1 = predict_views_per_day(title, model, days_old=1)
        print(f"Title: {title}")
        print(f"  Views/day (day 1):     {vpd_day1:>12,.0f}")
        for days in PROJECTION_DAYS:
            vpd_at_horizon = predict_views_per_day(title, model, days_old=days)
            projected_total = vpd_at_horizon * days
            print(f"  Projected views by {days}d: {projected_total:>12,.0f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
