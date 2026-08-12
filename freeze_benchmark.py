import hashlib
import os

import config
import scrape
import thumbnails


def _split_for(video_id):
    bucket = int(hashlib.sha256(str(video_id).encode()).hexdigest(), 16) % 100
    if bucket < config.TRAIN_SPLIT_MAX:
        return "train"
    if bucket < config.VAL_SPLIT_MAX:
        return "val"
    return "test"


def main():
    df = scrape.load_cached(config.DATA_PATH)
    if df.empty:
        raise SystemExit(f"No data at {config.DATA_PATH} - run scrape.py first.")

    df = df.copy()
    df["split"] = df["video_id"].apply(_split_for)

    os.makedirs(os.path.dirname(config.BENCHMARK_PATH) or ".", exist_ok=True)
    df.to_csv(config.BENCHMARK_PATH, index=False)

    counts = df["split"].value_counts()
    print(f"Froze {len(df)} rows from {config.DATA_PATH} -> {config.BENCHMARK_PATH}")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counts.get(split, 0)}")

    print("Computing thumbnail features (downloads cached to data/thumbnails/)...")
    thumb_feats = thumbnails.compute_features_for_df(df)
    thumb_feats.insert(0, "video_id", df["video_id"].values)
    thumb_feats.to_csv(config.THUMBNAIL_FEATURES_PATH, index=False)
    print(f"Saved thumbnail features -> {config.THUMBNAIL_FEATURES_PATH}")


if __name__ == "__main__":
    main()
