import sys

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from videodata import engineer_title_features

MODEL_PATH = "youtube_view_count_predictor.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
FEATURE_MEDIANS_PATH = "feature_medians.pkl"

SAMPLE_TITLES = [
    "How to train a neural network",
    "Top 10 programming languages in 2024",
    "Beginner's guide to machine learning",
    "Best practices for software development",
    "Advanced Python tutorials",
]


def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        feature_medians = joblib.load(FEATURE_MEDIANS_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run videodata.py first to train and save the model.")
        sys.exit(1)
    return model, vectorizer, feature_medians


def predict_views_per_day(title, model, vectorizer, feature_medians):
    tfidf = vectorizer.transform([title])
    title_feats = engineer_title_features(title)
    # Use saved medians for duration and channel_follower_count
    extra = np.array([title_feats + list(feature_medians[len(title_feats):])], dtype=float)
    X = hstack([tfidf, csr_matrix(extra)])
    return float(np.expm1(model.predict(X)[0]))


def main():
    model, vectorizer, feature_medians = load_artifacts()

    titles = sys.argv[1:] if len(sys.argv) > 1 else SAMPLE_TITLES

    for title in titles:
        vpd = predict_views_per_day(title, model, vectorizer, feature_medians)
        print(f"Title: {title}")
        print(f"  Views/day:        {vpd:>12,.0f}")
        print(f"  Projected 30d:    {vpd * 30:>12,.0f}")
        print(f"  Projected 365d:   {vpd * 365:>12,.0f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
