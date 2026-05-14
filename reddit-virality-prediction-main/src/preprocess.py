"""Preprocess Reddit posts for Doc2Vec and machine learning."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reddit_posts_raw.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Lowercase text and remove URLs, punctuation, and extra spaces."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text into simple word tokens."""
    return [token for token in text.split() if len(token) > 1]


def add_viral_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define viral posts relative to each subreddit.

    A post is viral if its score is in the top 10% within its subreddit sample.
    This is better than using a global threshold because subreddits vary greatly in size.
    """
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce").fillna(0)

    thresholds = df.groupby("subreddit")["score"].transform(lambda x: x.quantile(0.90))
    df["viral"] = (df["score"] >= thresholds).astype(int)

    # If a subreddit has too few posts or all scores are the same, the threshold can label too many.
    # The classification model can still run, but this column gives a reasonable relative signal.
    return df


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Combine title and body text. Some Reddit posts have empty selftext, so title is essential.
    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")
    df["post_text"] = (df["title"] + " " + df["selftext"]).str.strip()
    df["clean_text"] = df["post_text"].apply(clean_text)
    df["tokens"] = df["clean_text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)
    df["title_length"] = df["title"].astype(str).str.len()
    df["has_body_text"] = (df["selftext"].astype(str).str.len() > 0).astype(int)
    df["has_url"] = df["url"].fillna("").astype(str).str.startswith("http").astype(int) if "url" in df else 0

    # Remove very short text rows because Doc2Vec needs meaningful tokens.
    df = df[df["token_count"] >= 3].copy()
    df = add_viral_label(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Reddit data.")
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "reddit_posts_processed.csv"))
    args = parser.parse_args()

    df = preprocess(Path(args.input), Path(args.output))
    print(f"Saved {len(df):,} processed posts to {args.output}")
    print(df[["subreddit", "title", "score", "viral", "token_count"]].head())


if __name__ == "__main__":
    main()
