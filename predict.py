import sys

import joblib
import numpy as np
import pandas as pd

import config
from features import NUMERIC_FEATURE_COLUMNS, engineer_title_features

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


def build_input_row(title):
    feats = engineer_title_features(title)
    feats["title"] = title
    # Unknown at title-only inference time; the pipeline's imputer fills
    # these from training medians.
    feats["log_days_old"] = np.nan
    feats["duration"] = np.nan
    feats["channel_follower_count"] = np.nan
    return pd.DataFrame([{col: feats[col] for col in ["title"] + NUMERIC_FEATURE_COLUMNS}])


def predict_views_per_day(title, model):
    row = build_input_row(title)
    return float(np.expm1(model.predict(row)[0]))


def main():
    model = load_model()
    titles = sys.argv[1:] if len(sys.argv) > 1 else SAMPLE_TITLES

    for title in titles:
        vpd = predict_views_per_day(title, model)
        print(f"Title: {title}")
        print(f"  Views/day:        {vpd:>12,.0f}")
        print(f"  Projected 30d:    {vpd * 30:>12,.0f}")
        print(f"  Projected 365d:   {vpd * 365:>12,.0f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
