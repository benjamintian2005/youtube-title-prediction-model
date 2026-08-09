# autoresearch: youtube title -> views/day predictor

The goal is a single number: **val log_mse** (log1p of views/day, mean squared error on the
frozen `benchmark/dataset.csv` `split == 'val'` rows), currently recorded in `champion.json`.
Lower is better. Your job each cycle is to try to beat it.

## Loop

1. **Read `champion.json`** for the current best val `log_mse` (and what achieved it).
2. **Pick one hypothesis.** A new title feature, a different model family or embedding, a
   hyperparameter change, a preprocessing tweak. Keep it to one idea per cycle - small,
   reviewable diffs, not a pile of simultaneous changes you can't attribute credit/blame to.
3. **Implement it** in the agent-editable surface: `features.py`, `train.py`, `config.py`,
   `requirements.txt`. New dependencies are fine (embeddings, other regressors, etc.) - just
   add them to `requirements.txt` when you use them.
4. **Run `python train.py`.** It trains every candidate in `build_candidates()` on
   `split == 'train'`, scores them on `split == 'val'` only, appends every candidate's metrics
   to `metrics.csv` (win or lose - that's the permanent experiment log), and writes the best
   one to `candidate.pkl`. It never touches `model.pkl`.
5. **Compare** `candidate.pkl`'s val `log_mse` (printed by `train.py`, also the last relevant
   row in `metrics.csv`) to `champion.json`.
   - **Better** -> promote it:
     ```bash
     cp candidate.pkl model.pkl
     ```
     Update `champion.json` (`log_mse`, `timestamp`, `git_commit` of the new HEAD, a short
     `note` describing the hypothesis and what changed). Commit the code change and
     `champion.json`/`model.pkl` are gitignored - promotion is reproducible from the commit
     hash, so committing the code change plus the new `champion.json` is enough:
     ```bash
     git add features.py train.py config.py requirements.txt champion.json
     git commit -m "describe the hypothesis and the val log_mse improvement"
     ```
   - **Not better** -> revert the code:
     ```bash
     git checkout -- features.py train.py config.py requirements.txt
     ```
     Leave `model.pkl`/`champion.json` alone. The `metrics.csv` row from step 4 stays - it's
     the record of what was tried and didn't work.
6. **Repeat**, but see **Pivoting** below before defaulting to "another hyperparameter tweak" -
   check whether it's time to change *what kind* of hypothesis you're trying.

## Pivoting

Hyperparameter/preprocessing tweaks (steps 1-6 above) are the cheapest cycle, but they have the
smallest ceiling. Empirically, in this project, changes to *what data or features exist* have
dwarfed changes to *how the existing data/features are tuned*:

- Growing the dataset 403 -> 4103 rows + adding a `category` feature: `log_r2` 0 -> ~0.59.
- Five straight cycles of TF-IDF/model hyperparameter tuning on top of that: `log_mse` moved
  ~0.3%, and 4 of 5 cycles were discards.

So: **track the last several cycles** (`metrics.csv` + `champion.json` history). If recent
cycles are mostly discards, or the kept improvements are tiny relative to what data/feature
changes have bought historically, stop tuning the current lever and pivot instead. Try, roughly
in this order (highest historical leverage first):

1. **More/better data.** Run `api_scrape.py` again (raise `MAX_RESULTS_PER_SEED` further, add
   new seed word topics to `SEED_WORDS`, or just re-run as-is if it's been a while), then
   `python freeze_benchmark.py` to start a new epoch. Re-freezing resets what `champion.json`'s
   metric is comparable to (see the note field convention in past commits) - that's expected,
   not a problem, just record it honestly in the new `champion.json`'s `note`.
2. **A feature axis that doesn't exist yet**, not just a tweak to an existing one (`category`
   was this kind of win). Thumbnail-derived features are the obvious next one (see the file's
   git history around when this was discussed for the shape of that idea: a thumbnail URL is
   already free from both scrapers' API payloads, but needs a new local cache and a real
   implementation - go build it if you pivot here, don't just leave it as a TODO).
3. **Model/hyperparameter tuning** (the default loop above) - use this once 1 and 2 are
   reasonably exhausted for now, or to fill a cycle while deciding on a bigger pivot.

**Structural pivots are in-scope to implement autonomously**, including new dependencies, new
scraper logic, or new caching layers (e.g. a thumbnail image/embedding cache under `data/`,
gitignored like `data/videos.csv`) - not just parameter changes. Hold them to the same bar as
everything else: test end-to-end before committing (imports cleanly, `predict.py` and
`check_test.py` still work, sane output on a few real examples), write a clear commit message,
and record what you did in the batch digest. The one hard stop: if a pivot needs something you
don't have access to (a new paid API key, a credential, a resource beyond what's already
configured), don't attempt it silently - note it as a blocked option in the digest instead.

**Still run in a bounded batch.** Pivoting to a bigger idea doesn't mean looping forever -
whatever fixed number of cycles or wall-clock budget was given for this run still applies; stop
at the end of it and report a full digest (what was tried, kept, discarded, and why), same as
any other batch.

## Guardrails

- **Don't touch `benchmark/dataset.csv` or re-freeze it as part of a routine hyperparameter
  cycle** - only as a deliberate **Pivoting** action (see above) when growing the data is the
  chosen lever, and only between cycles, never mid-cycle. `predict.py`'s input contract
  (title-only CLI, falls back to imputed/unknown for anything else) shouldn't change casually
  either - if a pivot needs it to accept new input (e.g. a thumbnail pivot letting a user pass
  an image), that's a deliberate, documented interface change, not a side effect.
- **Never evaluate against `split == 'test'` during the loop.** `check_test.py` exists
  specifically so the held-out test set doesn't get implicitly optimized against every
  iteration. Run it only occasionally (e.g. every ~10 kept improvements) as a sanity check
  that val gains are generalizing, not overfit to the val set itself.
- **One hypothesis per iteration.** If a change doesn't help, you want to know which change,
  not "which combination of three changes."
- **`predict.py` should keep working unchanged.** It only depends on `features.py`'s exported
  contract (`NUMERIC_FEATURE_COLUMNS`, `engineer_title_features`, `build_preprocessor`) plus
  whatever `model.pkl` pipeline is currently promoted - as long as any new features you add
  are produced by `engineer_title_features`/`build_preprocessor` (not something only available
  at training time, like real `duration`/`channel_follower_count`, which are legitimately
  unknown for an unpublished title and get imputed), `predict.py` doesn't need touching.
- **Keep a run's wall-clock reasonable** (a couple minutes is a good target) so grid searches
  or embedding downloads don't balloon each cycle.
