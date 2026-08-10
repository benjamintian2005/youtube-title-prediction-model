from concurrent.futures import ThreadPoolExecutor
import io
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

CACHE_DIR = "data/thumbnails"
DOWNLOAD_TIMEOUT = 8
RESIZE_TO = (64, 64)  # small + fixed so stats are cheap and comparable regardless of source resolution
MAX_WORKERS = 20  # downloads are I/O-bound; a thread pool is the cheap win here

THUMBNAIL_FEATURE_COLUMNS = [
    "thumb_brightness", "thumb_saturation", "thumb_contrast",
    "thumb_edge_density", "thumb_warmth",
]

_NAN_FEATURES = {col: np.nan for col in THUMBNAIL_FEATURE_COLUMNS}


def _cache_path(video_id):
    return os.path.join(CACHE_DIR, f"{video_id}.jpg")


def _download(video_id, url):
    path = _cache_path(video_id)
    if os.path.exists(path):
        return path
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        if resp.status_code != 200 or not resp.content:
            return None
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception:
        return None


def _extract_features(path):
    try:
        img = Image.open(path)
    except Exception:
        return dict(_NAN_FEATURES)
    return _extract_features_from_image(img)


def _extract_features_from_image(img):
    try:
        img = img.convert("RGB").resize(RESIZE_TO)
    except Exception:
        return dict(_NAN_FEATURES)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    saturation = hsv[..., 1]

    dy, dx = np.gradient(gray)
    edge_density = np.sqrt(dx ** 2 + dy ** 2).mean()

    return {
        "thumb_brightness": float(gray.mean()),
        "thumb_saturation": float(saturation.mean()),
        "thumb_contrast": float(gray.std()),
        "thumb_edge_density": float(edge_density),
        "thumb_warmth": float(r.mean() - b.mean()),
    }


def get_features(video_id, url):
    path = _download(video_id, url)
    if path is None:
        return dict(_NAN_FEATURES)
    return _extract_features(path)


def features_from_source(source):
    """Compute thumbnail features from an arbitrary local file path or URL, for
    predict.py's --thumbnail option (a candidate thumbnail for a not-yet-published
    title). Unlike get_features(), this doesn't go through the video_id-keyed
    CACHE_DIR - there's no video_id yet, and the point is to score whatever image
    the caller hands in, not a scraped training example. Bad path/URL -> NaN
    features, same imputation-friendly contract as the rest of this module."""
    try:
        if source.startswith("http://") or source.startswith("https://"):
            resp = requests.get(source, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        else:
            img = Image.open(source)
    except Exception:
        return dict(_NAN_FEATURES)
    return _extract_features_from_image(img)


def compute_features_for_df(df, video_id_col="video_id", url_col="thumbnail_url"):
    """Downloads (cached under CACHE_DIR) + extracts thumbnail features for every row.
    Missing/failed thumbnails get NaN features, imputed downstream like duration/
    channel_follower_count - never raises on a single bad row."""
    pairs = list(zip(df[video_id_col], df[url_col]))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda p: get_features(*p), pairs))
    return pd.DataFrame(results, index=df.index)
