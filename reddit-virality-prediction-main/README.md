# Reddit Virality Prediction with Doc2Vec and t-SNE

This project collects **real Reddit post data** using Reddit public JSON API endpoints, converts each post into a numerical vector using **Doc2Vec**, and visualizes the text embeddings using **t-SNE** to evaluate the quality of clusters.

The project also keeps the original machine learning goal: predicting whether a Reddit post is likely to be viral based on early engagement and text-based features.

## Professor requirement covered

> Get real data. Convert the post into a vector. Use t-SNE to visualize the embedding, to see the quality of clusters.

This project satisfies that requirement with the following pipeline:

```text
Reddit public API → real posts → text cleaning → Doc2Vec vectors → t-SNE visualization → cluster quality analysis → virality classifier
```

## Important note about the Reddit API

This version uses Reddit's public JSON endpoints, for example:

```text
https://www.reddit.com/r/technology/hot.json
```

This means **no Reddit developer credentials are required**. The data is still collected from Reddit through an API-style JSON endpoint.

## Project structure

```text
reddit-virality-prediction/
├── data/
│   ├── raw/                         # Raw Reddit API data
│   └── processed/                   # Cleaned data and embeddings
├── notebooks/
│   ├── reddit_virality_simple.ipynb # Original notebook
│   └── reddit_doc2vec_tsne_pipeline.ipynb
├── outputs/
│   ├── figures/                     # t-SNE plots, confusion matrices, feature importance
│   ├── models/                      # Saved Doc2Vec and ML models
│   └── tables/                      # Metrics and cluster summaries
├── src/
│   ├── collect_reddit_public_api.py
│   ├── preprocess.py
│   ├── doc2vec_tsne.py
│   ├── train_classifier.py
│   └── run_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

Create a virtual environment if desired:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the complete project

The easiest option is to run the complete pipeline:

```bash
python src/run_pipeline.py
```

This will:

1. Collect real Reddit posts.
2. Clean and preprocess text.
3. Train Doc2Vec.
4. Convert posts into vectors.
5. Use t-SNE to reduce vectors to 2 dimensions.
6. Cluster the embeddings.
7. Train a virality prediction model.
8. Save figures, models, and tables.

## Run each step manually

### 1. Collect real Reddit posts

```bash
python src/collect_reddit_public_api.py --limit 100 --subreddits technology MachineLearning dataisbeautiful science gaming movies worldnews todayilearned
```

Outputs:

```text
data/raw/reddit_posts_raw.csv
```

### 2. Preprocess posts

```bash
python src/preprocess.py
```

Outputs:

```text
data/processed/reddit_posts_processed.csv
```

### 3. Train Doc2Vec and create t-SNE visualization

```bash
python src/doc2vec_tsne.py
```

Outputs:

```text
data/processed/doc2vec_vectors.csv
outputs/figures/tsne_by_subreddit.png
outputs/figures/tsne_by_viral_status.png
outputs/figures/tsne_by_cluster.png
outputs/tables/cluster_quality_summary.csv
outputs/models/doc2vec_model.model
```

### 4. Train classifier

```bash
python src/train_classifier.py
```

Outputs:

```text
outputs/tables/classification_metrics.csv
outputs/figures/confusion_matrix.png
outputs/figures/roc_curve.png
outputs/models/virality_classifier.joblib
```

## What is Doc2Vec?

Doc2Vec is a Natural Language Processing technique that converts each document or post into a numerical vector. In this project, each Reddit post becomes a vector such as:

```text
[0.14, -0.33, 0.72, ..., 0.09]
```

Posts with similar language and topics should have vectors that are close to each other.

## What is t-SNE?

The Doc2Vec vectors may have 100 dimensions, which humans cannot directly visualize. t-SNE reduces those vectors into 2 dimensions so that each post can be plotted as a point on a graph.

If the embeddings are meaningful, posts from similar communities or topics should appear close together.

## How virality is defined

A post is labeled as viral if its score is in the top 10% of scores within its subreddit sample.

```text
viral = 1 if score >= 90th percentile score for that subreddit
viral = 0 otherwise
```

This definition avoids comparing small subreddits directly against very large subreddits.

## Suggested explanation for presentation

This project extends a Reddit virality classifier by adding a real-data text embedding pipeline. Reddit posts are collected through public JSON API endpoints. The post title and body are cleaned and then converted into dense numerical representations using Doc2Vec. These vectors are visualized with t-SNE to evaluate whether posts with similar topics naturally form clusters. The same embeddings are then used as machine learning features for predicting viral posts.

## Reproducibility notes

- Reddit content changes constantly, so results may vary depending on the day and time the data is collected.
- If Reddit temporarily blocks a request, wait a few minutes and run again.
- The script uses a custom User-Agent and small delays between requests to avoid excessive traffic.
