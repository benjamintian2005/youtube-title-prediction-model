# youtube title → view predictor

predicts how many views a youtube video might get based on its title (plus a few side signals). trained on live data scraped with yt-dlp — no api key needed.

this is set up as an **autoresearch**-style project (see [karpathy/autoresearch](https://github.com/karpathy/autoresearch)): a frozen, comparable benchmark; one north-star metric (val `log_mse`); a narrow surface a coding agent iterates on; and a mechanical keep/discard loop. `program.md` is the actual instructions for running that loop.

## how it works

1. `scrape.py` searches youtube for a list of seed keywords via yt-dlp, dedupes by video id, and accumulates results into `data/videos.csv` (safe to re-run — it merges with what's already cached instead of overwriting). gitignored — this is the raw, ever-growing cache. `api_scrape.py` is an optional alternative/supplement: same candidate discovery (still free, via yt-dlp), but fetches metadata through the YouTube Data API v3 instead of scraping video pages one at a time — faster, and reliably fills in `duration`/`channel_follower_count` where yt-dlp sometimes can't. Needs a `YOUTUBE_API_KEY` env var; see "optional: YouTube Data API" below. Writes into the same `data/videos.csv` cache.
2. `freeze_benchmark.py` takes a snapshot of `data/videos.csv` and assigns every row to `train`/`val`/`test` via a stable hash of its video id, writing `benchmark/dataset.csv` (committed to git). This is what makes experiments comparable to each other — it's run deliberately, occasionally, not on every training run. Re-run it when you want a new "research epoch" with more scraped data.
3. `train.py` loads `benchmark/dataset.csv`, drops videos scraped too soon after upload (view counts there are dominated by initial spikes, not steady-state velocity), builds title features (TF-IDF + length/punctuation/clickbait-pattern stats) plus `duration`/`channel_follower_count`/video age, fits candidates (a median baseline, Ridge, gradient boosting — see `build_candidates()`) on `split == 'train'`, and scores them on `split == 'val'` only. Every candidate's metrics get appended to `metrics.csv` (the permanent experiment log), and the best one is saved to `candidate.pkl`. It never touches `model.pkl`.
4. Promoting a candidate to champion (copying `candidate.pkl` → `model.pkl`, updating `champion.json`) only happens when it beats the current champion's val `log_mse` — see `program.md` for the exact loop.
5. `predict.py` loads `model.pkl` (the current champion) and predicts **views/day** for a given title (rather than raw views, so newer videos aren't penalized relative to older ones that have had more time to accumulate views).
6. `check_test.py` reports the champion's metric on `split == 'test'` — the held-out set that's deliberately *not* touched during iteration, so it stays a valid check that val improvements are generalizing rather than overfit to the val set.

## setup

```bash
pip install -r requirements.txt
```

## usage

get data (first time, or to grow the cache):
```bash
python scrape.py
```

freeze a benchmark (first time, or to start a new research epoch):
```bash
python freeze_benchmark.py
```

run the optimization loop — either manually yourself, or point a coding agent at `program.md` and let it iterate:
```bash
python train.py          # trains + scores candidates on train/val, writes candidate.pkl
# ... compare candidate.pkl's val log_mse to champion.json, promote if it wins (see program.md) ...
python check_test.py     # occasional sanity check against the held-out test set
```

predict titles with the current champion:
```bash
python predict.py                          # runs sample titles
python predict.py "my video title here"    # predict a specific title
```

### optional: YouTube Data API

`scrape.py` needs no api key. If you want faster/more complete scraping, create a key at the [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (enable "YouTube Data API v3" first — free tier is 10,000 quota units/day, and `api_scrape.py` only spends ~1 unit per 50 videos), then either:

```bash
# PowerShell
$env:YOUTUBE_API_KEY = "your-key-here"
```
or drop it in a local `.env` file (gitignored) as `YOUTUBE_API_KEY=your-key-here` — `api_scrape.py` will pick it up automatically if `python-dotenv` is installed.

```bash
python api_scrape.py
```

output looks like:
```
Title: how to make ramen at home
  Views/day (day 1):         4,120
  Projected views by 30d:   57,200
  Projected views by 365d: 452,600
```

## notes

- predictions are rough — title is a weak signal on its own. channel size, thumbnail, and upload timing matter way more. `duration`/`channel_follower_count` are used as features when training but are unknown at title-only prediction time, so `predict.py` falls back to training medians for them.
- video age (`log_days_old`) turned out to be the strongest feature in the trained model — views/day decays sharply as a video ages, so `predict.py` queries the model separately at each projection horizon (day 1, 30, 365) instead of scaling a single point-estimate, otherwise the 30d/365d numbers would silently assume a constant daily rate.
- `metrics.csv` rows from before the autoresearch refactor used an ad hoc, non-frozen train/test split and aren't directly comparable to rows logged since — the frozen `benchmark/dataset.csv` and val-based scoring started with the first `champion.json`.
- to get more training data, add seed words to `SEED_WORDS` in `config.py`, or just re-run `python scrape.py` periodically — the cache accumulates and dedupes automatically. Remember to re-run `freeze_benchmark.py` deliberately when you want that new data reflected in the benchmark.
