import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

# Cargar datos
data = pd.read_csv("archivov4.csv")

from fastapi import HTTPException
import pandas as pd

# Supongamos que 'data' es el DataFrame cargado con los datos de películas

@app.get("/director/{nombre_director}")
def get_director(nombre_director: str):
    try:
        # Filtrar las películas dirigidas por el director
        director_data = data[data['director'].str.contains(nombre_director, case=False, na=False)]
        
        # Verificar si el director tiene registros en el dataset
        if director_data.empty:
            raise HTTPException(status_code=404, detail="Director no encontrado")
        
        # Calcular el retorno total del director
        retorno_total = director_data['return'].sum()

        # Crear una lista de películas con la información solicitada
        peliculas_info = [
            {
                "nombre_pelicula": row['title'],
                "fecha_lanzamiento": row['release_year'],
                "retorno_individual": row['return'],
                "costo": row['budget'],
                "ganancia": row['budget'] * row['return']
            }
            for _, row in director_data.iterrows()
        ]
        
        # Devolver los resultados
        return {
            "nombre_director": nombre_director,
            "retorno_total": retorno_total,
            "peliculas": peliculas_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))






data = data[['title', 'overview']].dropna()  # Mantener solo las columnas de título y resumen, eliminando nulos

# Convertir títulos a minúsculas para comparación insensible a mayúsculas
data['title_lower'] = data['title'].str.lower()

# Crear directorio para guardar gráficos
os.makedirs("graphs", exist_ok=True)

# Exploración de datos y generación de gráficos
def generate_wordcloud():
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data['title']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras de títulos")
    path = "graphs/wordcloud.png"
    plt.savefig(path)
    plt.close()
    return path

def generate_histogram():
    data['overview_length'] = data['overview'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(data['overview_length'], bins=30, kde=True)
    plt.title("Distribución de la longitud de los resúmenes")
    plt.xlabel("Número de palabras")
    plt.ylabel("Frecuencia")
    path = "graphs/histogram.png"
    plt.savefig(path)
    plt.close()
    return path

# Optimización de recomendación con pre-cálculo de TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'].fillna(""))

app = FastAPI()

# Endpoint para mostrar la nube de palabras
@app.get("/wordcloud/")
async def wordcloud():
    path = generate_wordcloud()
    return FileResponse(path, media_type="image/png")

# Endpoint para mostrar el histograma
@app.get("/histogram/")
async def histogram():
    path = generate_histogram()
    return FileResponse(path, media_type="image/png")

# Función de recomendación optimizada
@app.get("/recommendation/")
async def recommendation(titulo: str):
    titulo = titulo.lower()

    if titulo not in data['title_lower'].values:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Obtener el índice de la película
    idx = data.index[data['title_lower'] == titulo].tolist()[0]
    
    # Calcular similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Obtener las películas más similares (excluyendo la misma)
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    recommendations = data.iloc[similar_indices]['title'].tolist()
    
    return {"recommendations": recommendations}
