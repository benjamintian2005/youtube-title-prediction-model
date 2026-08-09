import os
import time
from datetime import datetime

import pandas as pd
import yt_dlp

import config

# Pace per-video requests to reduce the odds of hitting YouTube's rate limit
# on the metadata fetch stage (it fetches one video page per candidate).
REQUEST_DELAY_SECONDS = 0.5

DATA_COLUMNS = [
    "video_id", "title", "view_count", "upload_date",
    "duration", "channel_follower_count", "scraped_at",
]


def _search_candidate_ids(seed_words, max_results):
    # extract_flat is fast (no per-video request) but current yt-dlp no
    # longer populates upload_date/channel_follower_count on flat search
    # results, so it's only used here to gather candidate video ids and
    # dedupe them across seed words before the slower per-video fetch below.
    flat_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    candidates = {}
    with yt_dlp.YoutubeDL(flat_opts) as ydl:
        for word in seed_words:
            try:
                results = ydl.extract_info(f"ytsearch{max_results}:{word}", download=False)
            except Exception as e:
                print(f"[warn] seed word '{word}' failed: {e}; skipping")
                continue
            for entry in (results or {}).get("entries", []) or []:
                video_id = entry.get("id")
                if video_id and video_id not in candidates:
                    candidates[video_id] = entry.get("title")
    return candidates


def fetch_data(seed_words, max_results=config.MAX_RESULTS_PER_SEED):
    scraped_at = datetime.today().strftime("%Y%m%d")
    candidates = _search_candidate_ids(seed_words, max_results)
    print(f"Found {len(candidates)} unique candidate videos, fetching full metadata...")

    data = []
    video_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(video_opts) as ydl:
        for i, video_id in enumerate(candidates):
            if i > 0:
                time.sleep(REQUEST_DELAY_SECONDS)
            try:
                entry = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
            except Exception as e:
                print(f"[warn] video '{video_id}' failed: {e}; skipping")
                continue
            title = entry.get("title")
            view_count = entry.get("view_count")
            upload_date = entry.get("upload_date")
            if title and view_count is not None and upload_date:
                data.append({
                    "video_id": video_id,
                    "title": title,
                    "view_count": int(view_count),
                    "upload_date": upload_date,
                    "duration": entry.get("duration"),
                    "channel_follower_count": entry.get("channel_follower_count"),
                    "scraped_at": scraped_at,
                })
    return data


def load_cached(path=config.DATA_PATH):
    if os.path.exists(path):
        return pd.read_csv(
            path,
            dtype={"video_id": str, "upload_date": str, "scraped_at": str},
        )
    return pd.DataFrame(columns=DATA_COLUMNS)


def save_and_merge(new_records, path=config.DATA_PATH):
    new_df = pd.DataFrame(new_records, columns=DATA_COLUMNS)
    combined = pd.concat([load_cached(path), new_df], ignore_index=True)
    combined = (
        combined.sort_values("scraped_at")
        .drop_duplicates(subset="video_id", keep="last")
        .reset_index(drop=True)
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def main():
    before = len(load_cached())
    print("Fetching data from YouTube...")
    records = fetch_data(config.SEED_WORDS)
    print(f"Collected {len(records)} raw rows this run")

    combined = save_and_merge(records)
    print(f"Cache: {before} -> {len(combined)} rows at {config.DATA_PATH}")


if __name__ == "__main__":
    main()
