import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from fastapi import FastAPI, HTTPException
from typing import List

# Crear la instancia de FastAPI
app = FastAPI()

# Cargar los datos y preprocesamiento
data = pd.read_csv('archivov4.csv')

# --- Análisis Exploratorio de los Datos (EDA) ---

# Resumen de los datos
print("Resumen de datos:")
print(data.info())
print(data.describe())

# Gráfico de distribuciones de puntuación de películas
plt.figure(figsize=(10, 6))
sns.histplot(data['vote_average'], bins=30, kde=True)  # Cambia 'rating' por el nombre real de la columna de puntuación si es diferente
plt.title("Distribución de puntuaciones de películas")
plt.xlabel("Puntuación")
plt.ylabel("Frecuencia")
plt.show()

# Nube de palabras con los títulos de las películas
all_titles = " ".join(data['title'].astype(str))  # Cambia 'title' por el nombre real de la columna de títulos si es diferente
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_titles)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de Palabras de Títulos de Películas")
plt.show()

# Preparar el vectorizador TF-IDF para los títulos de películas
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data['title'].astype(str))  # Cambia 'title' si es necesario

# Calcular la matriz de similitud del coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Función de recomendación
def get_recommendations(titulo: str) -> List[str]:
    # Encuentra el índice de la película en el dataset
    idx = data[data['title'].str.lower() == titulo.lower()].index
    if idx.empty:
        return []
    
    idx = idx[0]
    # Obtener las puntuaciones de similitud de la película con las demás
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Ordenar las películas en base a la similitud y seleccionar las 5 más similares
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Ignorar la primera ya que es la misma película
    
    # Obtener los títulos de las películas recomendadas
    recommended_titles = [data['title'].iloc[i[0]] for i in sim_scores]
    return recommended_titles

# --- Crear el endpoint para la API ---

@app.get("/recomendacion/{titulo}", response_model=List[str])
async def recomendacion(titulo: str):
    recommendations = get_recommendations(titulo)
    if not recommendations:
        raise HTTPException(status_code=404, detail=f"No se encontró la película: {titulo}")
    return recommendations
