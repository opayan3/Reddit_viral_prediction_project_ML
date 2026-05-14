"""Train Doc2Vec, convert posts to vectors, visualize with t-SNE, and evaluate clusters."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
for directory in [PROCESSED_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def parse_tokens(value) -> List[str]:
    """Safely parse token lists saved as strings in CSV."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return str(value).split()


def train_doc2vec(token_lists: List[List[str]], vector_size: int, epochs: int) -> Doc2Vec:
    documents = [TaggedDocument(words=tokens, tags=[str(i)]) for i, tokens in enumerate(token_lists)]
    model = Doc2Vec(
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4,
        dm=1,
        epochs=epochs,
        seed=42,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model


def make_tsne(vectors: np.ndarray, perplexity: int) -> np.ndarray:
    """Reduce vectors to two dimensions using t-SNE."""
    n_samples = vectors.shape[0]
    # t-SNE perplexity must be smaller than number of samples.
    safe_perplexity = max(5, min(perplexity, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    return tsne.fit_transform(vectors)


def plot_scatter(df: pd.DataFrame, color_column: str, title: str, output_path: Path) -> None:
    """Create a simple t-SNE scatterplot colored by a categorical column."""
    plt.figure(figsize=(11, 8))
    categories = df[color_column].astype(str).fillna("Unknown").unique()

    for category in categories:
        subset = df[df[color_column].astype(str) == category]
        plt.scatter(subset["tsne_x"], subset["tsne_y"], label=str(category), alpha=0.70, s=25)

    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    if len(categories) <= 15:
        plt.legend(markerscale=1.5, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Doc2Vec + t-SNE for Reddit posts.")
    parser.add_argument("--input", default=str(PROCESSED_DIR / "reddit_posts_processed.csv"))
    parser.add_argument("--vector-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["tokens_parsed"] = df["tokens"].apply(parse_tokens)
    df = df[df["tokens_parsed"].apply(len) >= 3].reset_index(drop=True)

    model = train_doc2vec(df["tokens_parsed"].tolist(), args.vector_size, args.epochs)
    model.save(str(MODEL_DIR / "doc2vec_model.model"))

    vectors = np.vstack([model.dv[str(i)] for i in range(len(df))])
    vector_columns = [f"doc2vec_{i}" for i in range(vectors.shape[1])]
    vector_df = pd.DataFrame(vectors, columns=vector_columns)

    scaled_vectors = StandardScaler().fit_transform(vectors)
    tsne_coords = make_tsne(scaled_vectors, args.perplexity)
    df["tsne_x"] = tsne_coords[:, 0]
    df["tsne_y"] = tsne_coords[:, 1]

    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_vectors)

    silhouette = silhouette_score(scaled_vectors, df["cluster"]) if len(df["cluster"].unique()) > 1 else np.nan

    output = pd.concat([df.reset_index(drop=True), vector_df], axis=1)
    output_path = PROCESSED_DIR / "doc2vec_vectors.csv"
    output.to_csv(output_path, index=False)

    cluster_summary = (
        output.groupby("cluster")
        .agg(
            posts=("id", "count"),
            avg_score=("score", "mean"),
            avg_comments=("num_comments", "mean"),
            viral_rate=("viral", "mean"),
            top_subreddit=("subreddit", lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )
    cluster_summary["silhouette_score_overall"] = silhouette
    cluster_summary.to_csv(TABLE_DIR / "cluster_quality_summary.csv", index=False)

    plot_scatter(output, "subreddit", "t-SNE Visualization of Doc2Vec Embeddings by Subreddit", FIG_DIR / "tsne_by_subreddit.png")
    plot_scatter(output, "viral", "t-SNE Visualization of Doc2Vec Embeddings by Viral Status", FIG_DIR / "tsne_by_viral_status.png")
    plot_scatter(output, "cluster", "t-SNE Visualization of Doc2Vec Embeddings by K-Means Cluster", FIG_DIR / "tsne_by_cluster.png")

    print(f"Saved Doc2Vec vectors to {output_path}")
    print(f"Saved t-SNE figures to {FIG_DIR}")
    print(f"Overall silhouette score: {silhouette:.4f}")
    print(cluster_summary)


if __name__ == "__main__":
    main()
