# Generic filler words ("the", "of", "is", ...) mostly surface whatever's
# already most-viewed for that term, which skews the scraped sample toward
# viral outliers and gives little topical/category diversity. Seeded with
# real topic phrases spanning YouTube's major content categories instead, so
# the "category" feature (see features.py) actually has variance to learn
# from and the sample isn't dominated by a handful of mega-viral clusters.
SEED_WORDS = [
    "tech review", "gaming highlights", "cooking recipe", "home workout",
    "personal finance tips", "diy woodworking", "makeup tutorial",
    "travel vlog", "true crime", "science experiment", "movie review",
    "comedy sketch", "guitar lesson", "car repair", "book review",
    "meditation guide", "language learning", "dog training",
    "gardening tips", "photography tutorial", "stock market news",
    "crypto explained", "football highlights", "basketball highlights",
    "yoga for beginners", "running tips", "coding tutorial", "ai news",
    "history documentary", "space exploration", "health tips",
    "parenting advice", "interior design", "fashion haul",
    "skincare routine", "camping guide", "fishing tips", "chess strategy",
    "anime review", "speedrun", "unboxing", "life hacks", "study tips",
    "career advice", "startup story", "real estate investing",
    "world news today", "standup comedy", "song cover", "dance tutorial",
    "magic tricks", "product review", "minecraft", "video essay",
]
# Each candidate now requires its own metadata fetch (see scrape.py) since
# yt-dlp's flat search no longer returns upload_date/channel_follower_count,
# so this stays modest for scrape.py (one metadata fetch per candidate, slow).
# api_scrape.py batches metadata fetches (cheap/fast), so this mostly just
# controls how deep into each seed word's search results to look - raising it
# surfaces candidates past the previous cutoff instead of re-finding the same
# top results on every run.
MAX_RESULTS_PER_SEED = 100

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
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
)

RIDGE_PARAM_GRID = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}

HGB_PARAM_GRID = {
    "model__max_iter": [100, 300],
    "model__max_depth": [3, 6, None],
    "model__learning_rate": [0.05, 0.1],
}
