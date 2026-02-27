from datetime import datetime

import joblib
import numpy as np
import yt_dlp
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

SEED_WORDS = [
    "the", "what", "you", "this", "that", "with", "in", "be",
    "and", "of", "to", "have", "on", "is", "for", "it", "win", "I", "ever",
]

MODEL_PATH = "youtube_view_count_predictor.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
FEATURE_MEDIANS_PATH = "feature_medians.pkl"


def fetch_data(seed_words, max_results=50):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }
    data = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for word in seed_words:
            results = ydl.extract_info(f"ytsearch{max_results}:{word}", download=False)
            for entry in results.get("entries", []):
                title = entry.get("title")
                view_count = entry.get("view_count")
                upload_date = entry.get("upload_date")
                if title and view_count is not None and upload_date:
                    data.append({
                        "title": title,
                        "view_count": int(view_count),
                        "upload_date": upload_date,
                        "duration": entry.get("duration"),
                        "channel_follower_count": entry.get("channel_follower_count"),
                    })
    return data


def engineer_title_features(title):
    alpha_chars = [c for c in title if c.isalpha()]
    return [
        len(title),
        len(title.split()),
        int(any(c.isdigit() for c in title)),
        sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1),
        int("?" in title),
        int("!" in title),
    ]


def build_feature_matrix(records, vectorizer, fit=False):
    titles = [r["title"] for r in records]
    if fit:
        tfidf = vectorizer.fit_transform(titles)
    else:
        tfidf = vectorizer.transform(titles)

    extra = np.array(
        [
            engineer_title_features(r["title"]) + [
                r["duration"] or np.nan,
                r["channel_follower_count"] or np.nan,
            ]
            for r in records
        ],
        dtype=float,
    )
    return tfidf, extra


def train_model(data):
    today = datetime.today()

    for r in data:
        days_old = max(1, (today - datetime.strptime(r["upload_date"], "%Y%m%d")).days)
        r["views_per_day"] = r["view_count"] / days_old

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf, extra = build_feature_matrix(data, vectorizer, fit=True)

    feature_medians = np.nanmedian(extra, axis=0)
    nan_indices = np.where(np.isnan(extra))
    extra[nan_indices] = np.take(feature_medians, nan_indices[1])

    X = hstack([tfidf, csr_matrix(extra)])
    y = np.log1p([r["views_per_day"] for r in data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE (log scale): {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    print(f"Best params: {grid_search.best_params_}")

    return best_model, vectorizer, feature_medians


def main():
    print("Fetching data from YouTube...")
    data = fetch_data(SEED_WORDS)
    print(f"Collected {len(data)} videos")

    print("Training model...")
    model, vectorizer, feature_medians = train_model(data)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(feature_medians, FEATURE_MEDIANS_PATH)
    print(f"Saved: {MODEL_PATH}, {VECTORIZER_PATH}, {FEATURE_MEDIANS_PATH}")


if __name__ == "__main__":
    main()
