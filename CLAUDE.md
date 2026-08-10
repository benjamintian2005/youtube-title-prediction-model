# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A YouTube title → views/day predictor, structured as an **autoresearch**-style project (see
[karpathy/autoresearch](https://github.com/karpathy/autoresearch)): a frozen, comparable
benchmark; one north-star metric (val `log_mse`); a narrow agent-editable surface; and a
mechanical keep/discard promotion loop.

**`program.md` is the authoritative loop spec for iterating on the model — read it in full
before making any modeling change.** It is not optional background reading; it defines the
exact procedure (hypothesis → implement → `train.py` → compare → promote-or-revert), the
guardrails, and the pivoting strategy. Do not improvise a different loop.

## Commands

```bash
pip install -r requirements.txt

python scrape.py              # grow data/videos.csv via yt-dlp (no API key needed)
python api_scrape.py          # alternative/supplement: YouTube Data API v3 (needs YOUTUBE_API_KEY)
python freeze_benchmark.py    # snapshot data/videos.csv -> benchmark/dataset.csv with train/val/test split (deliberate, occasional)

python train.py               # train all candidates, score on val, log to metrics.csv, write candidate.pkl
python check_test.py          # score champion on held-out test split (run rarely, not every cycle)
python predict.py                        # sample predictions from current champion (model.pkl)
python predict.py "some title here"      # predict a specific title
python predict.py "title A" "title B"    # 2+ titles -> also ranks them against each other (primary intended use, see README)

python plot_progress.py       # regenerate progress.png from champion.json's git history (after a promotion)
```

There is no test suite/linter in this repo; `check_test.py` is a one-off metric check, not a
unit test runner.

## Architecture / data flow

```
scrape.py / api_scrape.py -> data/videos.csv (gitignored, ever-growing cache)
                                     |
                         freeze_benchmark.py (deliberate, occasional)
                                     v
                      benchmark/dataset.csv (committed; has 'split' column)
                                     |
                                train.py
                       (filters, builds features, fits candidates,
                        scores on val only, appends to metrics.csv)
                                     v
                            candidate.pkl (gitignored)
                                     |
                     manual compare vs champion.json's log_mse
                          (see "Promotion loop" below)
                                     v
                    model.pkl + champion.json (gitignored, if promoted)
                                     |
                                predict.py
```

- **`config.py`** — all tunable constants: `SEED_WORDS`, split thresholds (`TRAIN_SPLIT_MAX`/
  `VAL_SPLIT_MAX`, hash-bucket based), `MIN_DAYS_OLD`, TF-IDF params, model hyperparameter grids,
  file paths. Start here when tuning.
- **`features.py`** — the feature contract. `engineer_title_features()` (title-only, deterministic
  string/regex stats) and `build_preprocessor()` (a `ColumnTransformer`: TF-IDF on `title` +
  numeric pipeline + categorical `category` one-hot) are what `predict.py` depends on. Any new
  feature must be derivable from title-only input at inference time (falls back to imputed/median
  for anything only known at scrape time, like `duration`/`channel_follower_count`).
- **`thumbnails.py`** — downloads/caches thumbnails into `data/thumbnails/` (gitignored) and
  extracts cheap image stats (brightness, saturation, contrast, edge density, warmth) rather than
  a vision embedding, exported as `THUMBNAIL_FEATURE_COLUMNS`. Also falls back to imputed values
  at title-only prediction time.
- **`train.py`** — loads the frozen benchmark, applies `MIN_DAYS_OLD` filtering and feature
  engineering *after* loading (so re-freezing isn't needed when filter/feature logic changes),
  fits every candidate from `build_candidates()` (`dummy` baseline, `ridge`, `hgb`) on
  `split == 'train'`, scores on `split == 'val'` only, appends every result (win or lose) to
  `metrics.csv`, and writes only the best (non-dummy) candidate to `candidate.pkl`. Never
  touches `model.pkl` — promotion is a separate, manual step.
- **`predict.py`** — loads `model.pkl` and predicts views/day at multiple horizons (day 1/30/365)
  by querying the model separately at each `log_days_old`, since `log_days_old` is the strongest
  feature and decay is not linear. Title is required; `--category`/`--duration`/`--followers`/
  `--thumbnail` are optional CLI flags for anything else you happen to know ahead of publishing
  (falls back to imputed medians / an "unknown" category bucket when omitted). Per permutation
  importance on val, `channel_follower_count` and `category` are the #1 and #4 most important
  features overall, so those two flags matter most for prediction accuracy. When called with 2+
  titles, `compare_titles()`/`print_comparison()` also rank them against each other (day-1
  views/day, with category/duration/followers/thumbnail held identical across all of them) — per
  the README, this ranked comparison is the tool's primary intended use, not a bonus feature: a
  single title's absolute prediction is soft (val `mdape` ~69%), but non-title noise cancels out
  of a same-context ranking instead of compounding into it.
- **`champion.json`** — records the current champion's metrics, `git_commit`, and a `note`
  explaining the hypothesis/change. `model.pkl`/`candidate.pkl`/`champion.json` are all gitignored
  — reproducibility comes from the committed code + `champion.json`'s `git_commit`, not from
  committing the binaries.
- **`metrics.csv`** — permanent, append-only experiment log (every candidate from every
  `train.py` run, kept or discarded). Rows from before the autoresearch refactor used a
  non-frozen split and aren't comparable to later rows.

## Promotion loop (see `program.md` for full detail)

1. Pick **one** hypothesis per cycle (feature, model family, hyperparameter, preprocessing).
2. Implement in the agent-editable surface: `features.py`, `train.py`, `config.py`,
   `requirements.txt`.
3. `python train.py` — compare `candidate.pkl`'s val `log_mse` to `champion.json`.
4. **Better** → `cp candidate.pkl model.pkl`, update `champion.json` (log_mse, timestamp,
   git_commit, note), commit code + `champion.json` (never the `.pkl` files), then run
   `python plot_progress.py` and commit `progress.png`.
5. **Not better** → `git checkout -- features.py train.py config.py requirements.txt` and leave
   `model.pkl`/`champion.json` alone. The discard's row stays in `metrics.csv`.

**Guardrails:**
- Never touch/re-freeze `benchmark/dataset.csv` mid-cycle — only as a deliberate pivot, between
  cycles.
- Never evaluate against `split == 'test'` during the loop — `check_test.py` is a rare sanity
  check only.
- One hypothesis per iteration — otherwise you can't attribute credit/blame.
- `predict.py`'s title-only input contract must keep working unchanged unless a pivot
  deliberately and explicitly changes it.
- Keep each `train.py` run to roughly a couple of minutes — don't let grid searches or embedding
  downloads balloon per cycle.

When several recent cycles are discards or gains are marginal relative to what data/feature
pivots have historically bought, stop tuning and pivot instead (more data via
`api_scrape.py`/`SEED_WORDS` + re-freeze, or a genuinely new feature axis) — see `program.md`'s
"Pivoting" section for the full reasoning and ordering.

## Optional: YouTube Data API

`api_scrape.py` needs a `YOUTUBE_API_KEY` (env var or gitignored `.env` file, picked up via
`python-dotenv` if installed). Free tier is 10,000 quota units/day; `api_scrape.py` spends ~1
unit per 50 videos.
