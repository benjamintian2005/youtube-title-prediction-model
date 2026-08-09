SEED_WORDS = [
    "the", "what", "you", "this", "that", "with", "in", "be",
    "and", "of", "to", "have", "on", "is", "for", "it", "win", "I", "ever",
]
# Each candidate now requires its own metadata fetch (see scrape.py) since
# yt-dlp's flat search no longer returns upload_date/channel_follower_count,
# so this is kept modest to keep a full scrape run to a few minutes.
MAX_RESULTS_PER_SEED = 30

DATA_PATH = "data/videos.csv"
METRICS_PATH = "metrics.csv"
MODEL_PATH = "model.pkl"

# Frozen benchmark: a snapshot of DATA_PATH with a `split` column assigned by
# a stable hash of video_id, so the partition survives new scraping and stays
# valid even if filtering logic (e.g. MIN_DAYS_OLD) changes later. Written by
# freeze_benchmark.py; regenerating it is a deliberate, occasional action, not
# something train.py does on its own. See program.md.
BENCHMARK_PATH = "benchmark/dataset.csv"
TRAIN_SPLIT_MAX = 70  # hash bucket 0-69 -> train
VAL_SPLIT_MAX = 85  # hash bucket 70-84 -> val, 85-99 -> test

# train.py always writes its best candidate here; MODEL_PATH (the deployed
# champion predict.py loads) is only updated by promoting a candidate that
# beats CHAMPION_PATH's recorded metric. See program.md.
CANDIDATE_PATH = "candidate.pkl"
CHAMPION_PATH = "champion.json"

# Videos scraped within this many days of upload have view counts dominated
# by initial spikes rather than steady-state velocity; drop them.
MIN_DAYS_OLD = 3

TFIDF_PARAMS = dict(
    max_features=2000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

RIDGE_PARAM_GRID = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}

HGB_PARAM_GRID = {
    "model__max_iter": [100, 300],
    "model__max_depth": [3, 6, None],
    "model__learning_rate": [0.05, 0.1],
}
