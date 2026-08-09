# youtube title → view predictor

predicts how many views a youtube video might get based on its title (plus a few side signals). trained on live data scraped with yt-dlp — no api key needed.

## how it works

1. `scrape.py` searches youtube for a list of seed keywords via yt-dlp, dedupes by video id, and accumulates results into `data/videos.csv` (safe to re-run — it merges with what's already cached instead of overwriting).
2. `train.py` loads the cached data, drops videos scraped too soon after upload (view counts there are dominated by initial spikes, not steady-state velocity), builds title features (TF-IDF + length/punctuation/clickbait-pattern stats) plus `duration`/`channel_follower_count`/video age, and compares a few models (a median baseline, Ridge regression, gradient boosting) via cross-validated grid search. The best model is saved as a single `model.pkl` pipeline, and its metrics are appended to `metrics.csv`.
3. `predict.py` loads `model.pkl` and predicts **views/day** for a given title (rather than raw views, so newer videos aren't penalized relative to older ones that have had more time to accumulate views).

## setup

```bash
pip install -r requirements.txt
```

## usage

scrape more data (optional standalone step — `train.py` will also scrape automatically if `data/videos.csv` doesn't exist yet):
```bash
python scrape.py
```

train the model (takes a few minutes; hits youtube if no cached data exists yet):
```bash
python train.py
```

predict titles:
```bash
python predict.py                          # runs sample titles
python predict.py "my video title here"    # predict a specific title
```

output looks like:
```
Title: how to make ramen at home
  Views/day:          1,240
  Projected 30d:     37,200
  Projected 365d:   452,600
```

## notes

- predictions are rough — title is a weak signal on its own. channel size, thumbnail, and upload timing matter way more. `duration`/`channel_follower_count` are used as features when training but are unknown at title-only prediction time, so `predict.py` falls back to training medians for them.
- `metrics.csv` keeps an append-only history of model comparisons (model name, params, CV/test metrics) across runs, so progress is trackable via git history.
- to get more training data, add seed words to `SEED_WORDS` in `config.py`, or just re-run `python scrape.py` periodically — the cache accumulates and dedupes automatically.
