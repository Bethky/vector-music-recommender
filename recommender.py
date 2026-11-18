import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# 1. Cargar los datos musicales
print("Cargando base de datos de canciones...")
df = pd.read_csv("dataset.csv")

# Eliminar duplicados por nombre de canción y artista para evitar repeticiones
df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)

# 2. Seleccionar las características matemáticas del audio (nuestro espacio vectorial 7D)
features = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "loudness",
]

# Eliminar filas que tengan algún dato vacío en esas columnas
df = df.dropna(subset=features).reset_index(drop=True)

# 3. Normalizar todas las variables al rango [0, 1] para que ninguna domine el cálculo
scaler = MinMaxScaler()
feature_matrix = scaler.fit_transform(df[features])


def recomendar_canciones(nombre_cancion, num_recomendaciones=5):
    # Buscar la canción en la base de datos (sin distinguir mayúsculas/minúsculas)
    coincidencias = df[
        df["track_name"].str.contains(nombre_cancion, case=False, na=False)
    ]

    if coincidencias.empty:
        print(
            f"\nNo se encontró ninguna canción con '{nombre_cancion}'. Prueba con otro título."
        )
        return

    # Tomar la versión más popular si hay varias con el mismo nombre
    cancion_base = coincidencias.sort_values(
        by="popularity", ascending=False
    ).iloc[0]
    idx = cancion_base.name

    print(
        f"\nCanción base elegida: '{cancion_base['track_name']}' - {cancion_base['artists']} (Género: {cancion_base['track_genre']})"
    )

    # 4. Calcular la Similitud del Coseno entre el vector de esta canción y todas las demás
    vector_objetivo = feature_matrix[idx].reshape(1, -1)
    similitudes = cosine_similarity(vector_objetivo, feature_matrix).flatten()

    # Ordenar de mayor a menor similitud (excluyendo la propia canción en la posición 0)
    indices_similares = similitudes.argsort()[::-1][1 : num_recomendaciones + 1]

    print(
        f"\nTop {num_recomendaciones} canciones más parecidas matemáticamente:\n"
    )
    for puesto, i in enumerate(indices_similares, start=1):
        titulo = df.loc[i, "track_name"]
        artista = df.loc[i, "artists"]
        porcentaje = similitudes[i] * 100
        print(f"{puesto}. {titulo} - {artista} ({porcentaje:.2f}% de similitud)")


# Bucle interactivo para probar canciones desde la terminal
if __name__ == "__main__":
    while True:
        consulta = input(
            "\nEscribe el nombre de una canción en inglés o español (o 'salir'): "
        )
        if consulta.lower() == "salir":
            break
        recomendar_canciones(consulta)