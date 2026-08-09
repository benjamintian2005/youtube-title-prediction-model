import os
import re
from datetime import datetime

import requests

import config
import scrape

# Optional: load YOUTUBE_API_KEY from a local .env file if python-dotenv is
# installed and one exists. Never required - falls back to a real env var.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_BASE = "https://www.googleapis.com/youtube/v3"
BATCH_SIZE = 50  # videos.list / channels.list max ids per request

# YouTube's category taxonomy is a small, stable, well-known set (videoCategories.list
# would fetch it live, but that's another quota-costing call for something that
# basically never changes) - names match yt-dlp's own `categories` field so
# scrape.py and api_scrape.py populate the same "category" column consistently.
CATEGORY_NAMES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
    "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
    "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
    "25": "News & Politics", "26": "Howto & Style", "27": "Education",
    "28": "Science & Technology", "29": "Nonprofits & Activism",
    "30": "Movies", "31": "Anime/Animation", "32": "Action/Adventure",
    "33": "Classics", "34": "Comedy", "35": "Documentary", "36": "Drama",
    "37": "Family", "38": "Foreign", "39": "Horror", "40": "Sci-Fi/Fantasy",
    "41": "Thriller", "42": "Shorts", "43": "Shows", "44": "Trailers",
}

_ISO8601_DURATION_RE = re.compile(
    r"^P(?:\d+D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
)


def _parse_iso8601_duration(value):
    match = _ISO8601_DURATION_RE.match(value or "")
    if not match:
        return None
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _chunks(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _get_api_key():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit(
            "YOUTUBE_API_KEY is not set. Create a key at "
            "https://console.cloud.google.com/apis/credentials (enable the "
            "'YouTube Data API v3' first), then set it as an environment "
            "variable (or put it in a local .env file as "
            "YOUTUBE_API_KEY=... if you have python-dotenv installed)."
        )
    return api_key


def _fetch_videos(video_ids, api_key):
    results = {}
    for batch in _chunks(video_ids, BATCH_SIZE):
        resp = requests.get(
            f"{API_BASE}/videos",
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(batch),
                "key": api_key,
            },
        )
        if resp.status_code != 200:
            print(f"[warn] videos.list failed ({resp.status_code}): {resp.text[:200]}")
            continue
        for item in resp.json().get("items", []):
            results[item["id"]] = item
    return results


def _fetch_channel_subscriber_counts(channel_ids, api_key):
    counts = {}
    for batch in _chunks(sorted(set(channel_ids)), BATCH_SIZE):
        resp = requests.get(
            f"{API_BASE}/channels",
            params={
                "part": "statistics",
                "id": ",".join(batch),
                "key": api_key,
            },
        )
        if resp.status_code != 200:
            print(f"[warn] channels.list failed ({resp.status_code}): {resp.text[:200]}")
            continue
        for item in resp.json().get("items", []):
            stats = item.get("statistics", {})
            if not stats.get("hiddenSubscriberCount"):
                counts[item["id"]] = stats.get("subscriberCount")
    return counts


def fetch_data(seed_words, max_results=config.MAX_RESULTS_PER_SEED, api_key=None):
    api_key = api_key or _get_api_key()
    scraped_at = datetime.today().strftime("%Y%m%d")

    # Candidate discovery stays on yt-dlp's free flat search - only the
    # per-video metadata fetch (the expensive/slow/incomplete part) moves to
    # the API, batched via videos.list (1 quota unit per 50 ids).
    candidates = scrape.search_candidate_ids(seed_words, max_results)
    print(f"Found {len(candidates)} unique candidate videos, fetching via YouTube Data API...")

    videos = _fetch_videos(list(candidates), api_key)
    print(f"videos.list returned metadata for {len(videos)}/{len(candidates)} candidates")

    channel_ids = [v["snippet"]["channelId"] for v in videos.values() if v.get("snippet", {}).get("channelId")]
    subscriber_counts = _fetch_channel_subscriber_counts(channel_ids, api_key)

    data = []
    for video_id, item in videos.items():
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        title = snippet.get("title")
        view_count = stats.get("viewCount")
        published_at = snippet.get("publishedAt")
        if not (title and view_count is not None and published_at):
            continue

        upload_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")
        channel_id = snippet.get("channelId")
        category_id = snippet.get("categoryId")

        data.append({
            "video_id": video_id,
            "title": title,
            "view_count": int(view_count),
            "upload_date": upload_date,
            "duration": _parse_iso8601_duration(content.get("duration")),
            "channel_follower_count": subscriber_counts.get(channel_id),
            "category": CATEGORY_NAMES.get(category_id, category_id),
            "scraped_at": scraped_at,
        })
    return data


def main():
    api_key = _get_api_key()
    before = len(scrape.load_cached())
    print("Fetching data via YouTube Data API v3...")
    records = fetch_data(config.SEED_WORDS, api_key=api_key)
    print(f"Collected {len(records)} rows this run")

    combined = scrape.save_and_merge(records)
    print(f"Cache: {before} -> {len(combined)} rows at {config.DATA_PATH}")


if __name__ == "__main__":
    main()
