import argparse
import sys

import joblib
import numpy as np
import pandas as pd

import config
import thumbnails
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


def build_input_row(title, days_old=None, category=None, duration=None,
                     channel_follower_count=None, thumbnail=None):
    feats = engineer_title_features(title)
    feats["title"] = title
    # days_old is provided per-call (see predict_views_per_day) since the
    # model learned a strong age/velocity relationship (see features.py)
    # and a single imputed age would silently bias every projection.
    feats["log_days_old"] = np.log1p(days_old) if days_old is not None else np.nan
    # duration/channel_follower_count/category/thumbnail are all unknowable
    # for a title with no video behind it yet, so they default to NaN and
    # the pipeline's imputer falls back to training medians / an "unknown"
    # category bucket. But category, channel_follower_count, and duration
    # are commonly known ahead of publishing (your own channel's subscriber
    # count, the category you're going to pick, a planned length), and a
    # candidate thumbnail is often already made - callers who have any of
    # these can pass them in for a materially more accurate prediction
    # (channel_follower_count and category are the model's #1 and #4 most
    # important features).
    feats["duration"] = duration if duration is not None else np.nan
    feats["channel_follower_count"] = (
        channel_follower_count if channel_follower_count is not None else np.nan
    )
    # channel_hist_log_vpd (a channel's own historical performance - see
    # train.py's _add_channel_history_feature) needs a channel_id lookup
    # against training data, which predict.py's CLI has no way to supply;
    # always falls back to the imputed median like duration.
    feats["channel_hist_log_vpd"] = np.nan
    feats["category"] = category if category is not None else np.nan
    if thumbnail:
        feats.update(thumbnails.features_from_source(thumbnail))
    else:
        for col in THUMBNAIL_FEATURE_COLUMNS:
            feats[col] = np.nan
    cols = ["title"] + NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    return pd.DataFrame([{col: feats[col] for col in cols}])


def predict_views_per_day(title, model, days_old=None, category=None, duration=None,
                           channel_follower_count=None, thumbnail=None):
    row = build_input_row(
        title, days_old=days_old, category=category, duration=duration,
        channel_follower_count=channel_follower_count, thumbnail=thumbnail,
    )
    return float(np.expm1(model.predict(row)[0]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict views/day for one or more YouTube titles. "
                    "category/duration/followers/thumbnail are all optional - "
                    "omit any you don't know yet and it falls back to the "
                    "training median (or an 'unknown' bucket for category)."
    )
    parser.add_argument(
        "titles", nargs="*",
        help="Title(s) to predict. Defaults to a few sample titles if omitted.",
    )
    parser.add_argument(
        "--category", default=None,
        help="YouTube category, e.g. 'Gaming' (see benchmark/dataset.csv for valid names).",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Planned video length in seconds.",
    )
    parser.add_argument(
        "--followers", type=float, default=None, dest="channel_follower_count",
        help="Channel subscriber count. This is the single strongest feature "
             "the model has - passing a real value here matters more than any "
             "other optional flag.",
    )
    parser.add_argument(
        "--thumbnail", default=None,
        help="Local file path or URL to a candidate thumbnail image.",
    )
    return parser.parse_args()


def compare_titles(titles, model, category=None, duration=None,
                    channel_follower_count=None, thumbnail=None):
    """Rank titles against each other with category/duration/channel_follower_count/
    thumbnail held identical across all of them (whether that's real values or the
    same imputed defaults - either way it's apples-to-apples), isolating title's
    effect. This is the model's main practical use: absolute forecasts are soft
    (val mdape ~69%, i.e. a typical single prediction is only right to within
    roughly 2-3x) but a same-context ranking is a much steadier signal, since
    systematic error in the imputed/shared context cancels out of the comparison
    rather than compounding into it. Returns (title, day-1 views/day) pairs sorted
    best-first.
    """
    scored = [
        (title, predict_views_per_day(
            title, model, days_old=1, category=category, duration=duration,
            channel_follower_count=channel_follower_count, thumbnail=thumbnail,
        ))
        for title in titles
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def print_comparison(titles, model, **extra):
    ranked = compare_titles(titles, model, **extra)
    best_title, best_vpd = ranked[0]
    print("Ranked comparison (day-1 views/day; category/duration/followers/thumbnail")
    print("held identical across all titles, so this isolates title's effect):")
    for rank, (title, vpd) in enumerate(ranked, start=1):
        pct_of_best = 100 * vpd / best_vpd if best_vpd else 0
        marker = "  <- predicted best" if rank == 1 else ""
        print(f"  {rank}. {pct_of_best:>4.0f}% of best  {vpd:>12,.0f}/day  {title}{marker}")
    print("  (title alone explains a modest share of view variance - treat this as a")
    print("   directional signal for A/B-ing titles, not a precise forecast for either.)")


def main():
    args = parse_args()
    model = load_model()
    titles = args.titles if args.titles else SAMPLE_TITLES
    extra = dict(
        category=args.category, duration=args.duration,
        channel_follower_count=args.channel_follower_count, thumbnail=args.thumbnail,
    )

    for title in titles:
        vpd_day1 = predict_views_per_day(title, model, days_old=1, **extra)
        print(f"Title: {title}")
        print(f"  Views/day (day 1):     {vpd_day1:>12,.0f}")
        for days in PROJECTION_DAYS:
            vpd_at_horizon = predict_views_per_day(title, model, days_old=days, **extra)
            projected_total = vpd_at_horizon * days
            print(f"  Projected views by {days}d: {projected_total:>12,.0f}")
        print("-" * 40)

    if len(titles) > 1:
        print_comparison(titles, model, **extra)


if __name__ == "__main__":
    main()
