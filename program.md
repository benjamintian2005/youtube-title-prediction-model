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
6. **Repeat.**

## Guardrails

- **Never touch `benchmark/dataset.csv`, `scrape.py`, or `predict.py`'s input contract** as
  part of a hypothesis. Those are the fixed harness - re-freezing the benchmark
  (`python freeze_benchmark.py`) is a deliberate, occasional, human-triggered action for
  starting a new research epoch (e.g. after a big new scrape), not something to do mid-loop.
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
