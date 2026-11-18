import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Vector Music Recommender", page_icon="🎵", layout="wide")

st.title("🎵 Vector-Based Music Recommender")
st.markdown("Recomendador musical basado en **álgebra lineal (vectores en $\\mathbb{R}^7$ y similitud del coseno)**.")

FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "loudness",
]

@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset.csv")
    df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    df["display_name"] = df["track_name"] + " — " + df["artists"]

    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(df[FEATURES])
    return df, matrix

df, feature_matrix = cargar_datos()

st.sidebar.header("⚙️ Parámetros del modelo")
num_recs = st.sidebar.slider("Número de recomendaciones", 3, 15, 5)
solo_joyas_ocultas = st.sidebar.checkbox("Modo 'Hidden Gems' (popularidad < 40)", value=False)

cancion_seleccionada = st.selectbox(
    "🔍 Busca o escribe el nombre de una canción:",
    options=df["display_name"].values,
    index=0,
)

if cancion_seleccionada:
    idx = int(df[df["display_name"] == cancion_seleccionada].index.tolist().pop(0))
    cancion_base = df.loc[idx]

    vector_objetivo = feature_matrix[idx].reshape(1, -1)
    similitudes = cosine_similarity(vector_objetivo, feature_matrix).flatten()

    resultados = df.copy()
    resultados["similarity"] = similitudes * 100
    resultados = resultados.drop(index=idx)

    if solo_joyas_ocultas:
        resultados = resultados[resultados["popularity"] < 40]

    top_recs = resultados.sort_values(by="similarity", ascending=False).head(num_recs)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader(f"Top {num_recs} canciones más parecidas")
        for rank, (_, row) in enumerate(top_recs.iterrows(), start=1):
            st.markdown(f"**{rank}. {row['track_name']}** — *{row['artists']}*")
            st.caption(
                f"Género: {row['track_genre']} | Popularidad: {row['popularity']}/100 | Similitud del coseno: **{row['similarity']:.2f}%**"
            )
            st.progress(min(int(row["similarity"]), 100))

    with col2:
        st.subheader("Comparación vectorial ($\\mathbb{R}^7$)")
        mejor_idx = int(top_recs.index.tolist().pop(0))

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=feature_matrix[idx].tolist(),
                theta=FEATURES,
                fill="toself",
                name=f"Base: {str(cancion_base['track_name'])[:20]}",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=feature_matrix[mejor_idx].tolist(),
                theta=FEATURES,
                fill="toself",
                name=f"Top 1: {str(df.loc[mejor_idx, 'track_name'])[:20]}",
            )
        )
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)