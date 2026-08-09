import re

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

FIRST_PERSON_RE = re.compile(r"\b(i|i'm|my|we|our|us)\b", re.IGNORECASE)
BRACKETS_RE = re.compile(r"[\[({].*?[\])}]")

NUMERIC_FEATURE_COLUMNS = [
    "title_len", "word_count", "has_digit", "caps_ratio",
    "has_question", "has_exclaim", "starts_with_digit", "has_brackets",
    "has_colon", "first_person_pronoun", "all_caps_word_count",
    "title_length_bucket", "log_days_old",
    "duration", "channel_follower_count",
]


def engineer_title_features(title):
    alpha_chars = [c for c in title if c.isalpha()]
    words = title.split()
    stripped = title.strip()
    length = len(title)
    return {
        "title_len": length,
        "word_count": len(words),
        "has_digit": int(any(c.isdigit() for c in title)),
        "caps_ratio": sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1),
        "has_question": int("?" in title),
        "has_exclaim": int("!" in title),
        "starts_with_digit": int(bool(stripped) and stripped[0].isdigit()),
        "has_brackets": int(bool(BRACKETS_RE.search(title))),
        "has_colon": int(":" in title),
        "first_person_pronoun": int(bool(FIRST_PERSON_RE.search(title))),
        "all_caps_word_count": sum(1 for w in words if len(w) > 1 and w.isupper()),
        "title_length_bucket": 0 if length < 30 else (1 if length <= 60 else 2),
    }


def build_preprocessor(sparse=True):
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # HistGradientBoostingRegressor requires dense input; Ridge is happy
    # with (and benefits from) sparse TF-IDF, so this is configurable.
    sparse_threshold = 0.3 if sparse else 0.0
    return ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(**config.TFIDF_PARAMS), "title"),
            ("num", numeric_pipeline, NUMERIC_FEATURE_COLUMNS),
        ],
        sparse_threshold=sparse_threshold,
    )


if __name__ == "__main__":
    for sample in [
        "How to train a neural network",
        "Top 10 programming languages in 2024",
        "I made $10,000 in ONE DAY (crazy results!)",
    ]:
        print(sample, "->", engineer_title_features(sample))
