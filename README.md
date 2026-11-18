# Vector-Based Music Recommender ($\mathbb{R}^7$)

An interactive content-based music recommendation web application built with **Python**, **Streamlit**, and **Scikit-learn**. It uses linear algebra and vector space modeling to find mathematically similar tracks across a dataset of over 114,000 songs.

## Mathematical Foundation

Each track in the dataset is represented as a 7-dimensional vector $\vec{v} \in \mathbb{R}^7$ based on its core audio features:
$$\vec{v} = (\text{danceability}, \text{energy}, \text{valence}, \text{tempo}, \text{acousticness}, \text{instrumentalness}, \text{loudness})$$

1. **Feature Normalization (`MinMaxScaler`):** Since audio features operate on different scales (e.g., *tempo* in BPM vs. *energy* between $0$ and $1$), all dimensions are normalized to the $[0, 1]$ interval so no single feature dominates the distance calculation.
2. **Cosine Similarity:** To measure how similar two tracks $\vec{A}$ and $\vec{B}$ are, the engine computes the cosine of the angle $\theta$ between their normalized vectors:
   $$\text{similarity}(\vec{A}, \vec{B}) = \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$
   A value close to $1$ ($100\%$) indicates that both tracks share nearly identical musical proportions and sonic atmosphere.

## Key Features

- **Real-Time Vector Search:** Instant song lookup across 114,000+ tracks with cosine similarity percentages.
- **Interactive 7D Radar Chart:** Visual comparison using **Plotly** showing how the base track and top recommendation overlap across all seven audio dimensions.
- **"Hidden Gems" Filter:** Option to filter out mainstream hits (`popularity < 40`) to discover lesser-known tracks with high mathematical similarity.
- **CLI & Web Interfaces:** Includes both a lightweight command-line script (`recommender.py`) and a full interactive web dashboard (`app.py`).

## Tech Stack

- **Language:** Python 3
- **Data Processing & Math:** Pandas, NumPy, Scikit-learn
- **Web Interface & Visualization:** Streamlit, Plotly
- **Dataset:** Spotify Tracks Dataset (Kaggle)

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd vector-music-recommender
   ```

2. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```
