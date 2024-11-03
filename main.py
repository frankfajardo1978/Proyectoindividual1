# Importar librerías necesarias
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar el archivo CSV
file_path = 'arcchivoprueba.csv'  # Cambia esto a la ruta del archivo si es necesario
df = pd.read_csv(file_path)

# Llenar valores nulos en columnas importantes
df['title'] = df['title'].fillna('')
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('[]')

# Crear una matriz TF-IDF basada en la sinopsis (overview)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Calcular la matriz de similitud de coseno
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Definir la función de recomendación
def recomendacion(titulo):
    # Obtener el índice de la película que coincide con el título
    idx = df.index[df['title'] == titulo].tolist()
    if not idx:
        return "Película no encontrada."
    
    idx = idx[0]

    # Obtener las puntuaciones de similitud de esa película con todas las demás
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenar las películas en base a la similitud de coseno
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Seleccionar los 5 títulos más similares
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    # Retornar los títulos de las películas
    return df['title'].iloc[movie_indices].tolist()

# Configurar la API con FastAPI
app = FastAPI()

# Estructura del cuerpo de la solicitud
class MovieTitle(BaseModel):
    titulo: str

@app.post("/recomendacion/")
def get_recomendacion(data: MovieTitle):
    return {"recomendaciones": recomendacion(data.titulo)}
