from fastapi import FastAPI, HTTPException
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Cargar los datos
file_path = '/mnt/data/archivov4.csv'  # Ruta del archivo subido
df = pd.read_csv(file_path)

# 1. Función para análisis exploratorio de datos (EDA)
@app.get("/eda")
def eda():
    # Analizar datos: algunas estadísticas básicas y visualizaciones de ejemplo
    eda_results = {
        "numero_de_filas": df.shape[0],
        "numero_de_columnas": df.shape[1],
        "columnas": df.columns.tolist(),
        "datos_nulos": df.isnull().sum().to_dict(),
        "descripcion": df.describe().to_dict()
    }
    return eda_results

# 2. Función de recomendación
# Procesamiento de texto para el sistema de recomendación
vectorizer = CountVectorizer()
count_matrix = vectorizer.fit_transform(df['title'])
cosine_sim = cosine_similarity(count_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    # Verificar si el título existe en el dataframe
    if title not in df['title'].values:
        raise ValueError("La película no se encuentra en la base de datos.")
    
    # Obtener el índice de la película
    idx = df[df['title'] == title].index[0]
    
    # Similaridades de la película con las demás
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Ordenar las películas según la similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtener los índices de las 5 películas más similares
    movie_indices = [i[0] for i in sim_scores[1:6]]
    
    # Retornar los títulos de las películas recomendadas
    return df['title'].iloc[movie_indices].tolist()

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo
