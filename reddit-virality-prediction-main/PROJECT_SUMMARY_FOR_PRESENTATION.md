# Project Summary for Presentation

This project predicts Reddit post virality using real Reddit data and Natural Language Processing.

## Main improvement

The original version used simulated data. The improved version collects real posts from Reddit using public JSON API endpoints. Each Reddit post is converted into a numerical vector using Doc2Vec. Then, t-SNE is used to visualize the high-dimensional embeddings in two dimensions and evaluate whether posts form meaningful clusters.

## Methods used

1. **Reddit public API collection**: collects titles, body text, subreddit, score, comments, posting time, and metadata.
2. **Text preprocessing**: combines title and body, lowercases text, removes punctuation and URLs, and tokenizes words.
3. **Doc2Vec**: converts each Reddit post into a dense numerical vector.
4. **t-SNE**: reduces the Doc2Vec vectors to two dimensions for visualization.
5. **K-Means clustering**: groups similar post embeddings and evaluates cluster quality using silhouette score.
6. **Random Forest classifier**: predicts whether a post is viral using text embeddings and metadata.

## Why this matters

The Doc2Vec model allows the project to use the actual semantic content of Reddit posts, not only numeric engagement features. The t-SNE plots help visually inspect whether posts from similar topics or communities are placed near one another. This makes the project more complete because it includes real data collection, text representation, unsupervised learning, visualization, and supervised machine learning.
