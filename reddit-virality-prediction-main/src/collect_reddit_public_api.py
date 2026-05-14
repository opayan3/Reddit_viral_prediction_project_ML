"""
Collect real Reddit post data from public Reddit JSON endpoints.

This script does not require Reddit API credentials. It uses URLs such as:
https://www.reddit.com/r/technology/hot.json

"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SUBREDDITS = [
    "technology",
    "MachineLearning",
    "dataisbeautiful",
    "science",
    "gaming",
    "movies",
    "worldnews",
    "todayilearned",
]

DEFAULT_LISTINGS = ["hot", "new", "top"]

HEADERS = {
    "User-Agent": "reddit-virality-ml-project/1.0 by student-research-project"
}


def fetch_listing(
    subreddit: str,
    listing: str = "hot",
    limit: int = 100,
    sleep_seconds: float = 1.0,
) -> List[Dict]:
    """Fetch posts from one subreddit/listing using Reddit public JSON endpoints."""
    posts: List[Dict] = []
    after: Optional[str] = None

    # Reddit listing endpoints usually return up to 100 posts per request.
    # We paginate until we reach the requested limit or Reddit returns no next page.
    while len(posts) < limit:
        remaining = min(100, limit - len(posts))
        url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"
        params = {"limit": remaining}
        if after:
            params["after"] = after

        response = requests.get(url, headers=HEADERS, params=params, timeout=30)

        if response.status_code == 429:
            print(f"Rate limited by Reddit for r/{subreddit}. Waiting 30 seconds...")
            time.sleep(30)
            continue

        if response.status_code != 200:
            print(f"Warning: failed r/{subreddit}/{listing} with status {response.status_code}")
            break

        payload = response.json()
        children = payload.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            data = child.get("data", {})
            posts.append(
                {
                    "id": data.get("id"),
                    "fullname": data.get("name"),
                    "subreddit": data.get("subreddit"),
                    "listing": listing,
                    "title": data.get("title", ""),
                    "selftext": data.get("selftext", ""),
                    "url": data.get("url", ""),
                    "permalink": "https://www.reddit.com" + data.get("permalink", ""),
                    "author": data.get("author"),
                    "score": data.get("score", 0),
                    "upvote_ratio": data.get("upvote_ratio"),
                    "num_comments": data.get("num_comments", 0),
                    "created_utc": data.get("created_utc"),
                    "over_18": data.get("over_18"),
                    "is_video": data.get("is_video"),
                    "stickied": data.get("stickied"),
                    "spoiler": data.get("spoiler"),
                    "locked": data.get("locked"),
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

        after = payload.get("data", {}).get("after")
        if not after:
            break

        time.sleep(sleep_seconds)

    return posts


def collect_posts(
    subreddits: Iterable[str],
    listings: Iterable[str],
    limit: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    """Collect posts from multiple subreddits and listings."""
    all_posts: List[Dict] = []

    tasks = [(sub, listing) for sub in subreddits for listing in listings]
    for subreddit, listing in tqdm(tasks, desc="Collecting Reddit data"):
        all_posts.extend(fetch_listing(subreddit, listing, limit, sleep_seconds))
        time.sleep(sleep_seconds)

    df = pd.DataFrame(all_posts)
    if df.empty:
        raise RuntimeError(
            "No posts were collected. Reddit may have blocked the request temporarily. "
            "Try again later or reduce the number of subreddits/listings."
        )

    df = df.drop_duplicates(subset=["id", "subreddit"]).reset_index(drop=True)

    if "created_utc" in df.columns:
        df["created_datetime"] = pd.to_datetime(df["created_utc"], unit="s", utc=True, errors="coerce")
        df["hour_posted"] = df["created_datetime"].dt.hour
        df["day_of_week"] = df["created_datetime"].dt.day_name()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect real Reddit data using public JSON endpoints.")
    parser.add_argument("--limit", type=int, default=100, help="Posts per subreddit per listing.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between requests in seconds.")
    parser.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS)
    parser.add_argument("--listings", nargs="+", default=DEFAULT_LISTINGS, choices=["hot", "new", "top", "rising"])
    parser.add_argument("--output", default=str(RAW_DIR / "reddit_posts_raw.csv"))
    args = parser.parse_args()

    df = collect_posts(args.subreddits, args.listings, args.limit, args.sleep)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df):,} real Reddit posts to {output_path}")
    print(df[["subreddit", "listing", "title", "score", "num_comments"]].head())


if __name__ == "__main__":
    main()
