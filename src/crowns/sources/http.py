"""Small HTTP helpers with retries and atomic file downloads."""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

TIMEOUT = 120
RETRIES = 4


def get(url: str, params: dict | None = None) -> requests.Response:
    """GET with retries and exponential backoff (2, 4, 8 s)."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last = e
            # Client errors other than rate limiting won't improve on retry.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                break
            if attempt < RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Request failed: {url} ({last})") from last


def download(url: str, dest: Path) -> Path:
    """Download to `dest` unless it already exists. Writes to a temp file first, so an
    interrupted download never leaves a partial file behind under the final name."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
            os.replace(tmp, dest)
            return dest
        except requests.RequestException as e:
            last = e
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Download failed: {url} ({last})") from last
