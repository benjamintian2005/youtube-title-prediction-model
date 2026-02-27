# youtube title → view predictor

predicts how many views a youtube video might get based on its title alone. trained on live data scraped with yt-dlp — no api key needed.

## how it works

1. scrapes youtube search results for a list of seed keywords using yt-dlp
2. trains a ridge regression model on TF-IDF title features + some basic title stats (length, caps ratio, punctuation, etc.)
3. predicts **views/day** rather than raw views to avoid penalizing newer videos

## setup

```bash
pip install yt-dlp scikit-learn numpy joblib scipy
```

## usage

train the model (takes a few minutes, hits youtube):
```bash
python videodata.py
```

predict titles:
```bash
python test.py                          # runs sample titles
python test.py "my video title here"    # predict a specific title
```

output looks like:
```
Title: how to make ramen at home
  Views/day:          1,240
  Projected 30d:     37,200
  Projected 365d:   452,600
```

## notes

- predictions are rough — title is a weak signal on its own. channel size, thumbnail, and upload timing matter way more
- `channel_follower_count` and `duration` are collected where available and used as features; missing values are imputed with training medians
- to get more training data, add seed words to `SEED_WORDS` in `videodata.py`
