import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import dask.dataframe as dd

# Cambia pandas por dask para carga diferida



# Inicialización del servidor FastAPI
app = FastAPI()

# Cargar y procesar el dataset una sola vez
try:
    # Cargar solo las columnas necesarias para reducir el uso de memoria
    df = dd.read_csv("archivov4.csv", usecols=["title", "overview"])
    if 'title' not in df.columns or 'overview' not in df.columns:
        raise ValueError("El dataset debe contener las columnas 'title' y 'overview'")

    # Rellenar valores nulos en la columna 'overview' para evitar errores
    df['overview'] = df['overview'].fillna('')

    # Vectorización y cálculo de similitud (precomputado)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
except Exception as e:
    print(f"Error al cargar o procesar el archivo: {e}")
    raise

# Modelo para manejar los datos entrantes (no se usa en esta versión, pero se mantiene para posibles futuras necesidades)
class Item(BaseModel):
    title: str
    rating: Optional[float] = None

@app.get("/")
def read_root():
    return {"message": "API activa y funcionando correctamente"}

# Función de recomendación optimizada
def recomendacion(titulo: str) -> List[str]:
    try:
        # Verificar si el título existe en el dataset
        if titulo not in df['title'].values:
            raise ValueError("La película no se encuentra en el dataset")

        # Índice de la película en el dataset
        idx = df.index[df['title'] == titulo][0]

        # Obtener índices de las 5 películas más similares, excluyendo la misma película
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:6]]
        
        # Retornar títulos de las películas recomendadas
        recomendaciones = df['title'].iloc[sim_indices].tolist()
        return recomendaciones
    except Exception as e:
        print(f"Error en la recomendación: {e}")
        raise HTTPException(status_code=500, detail="Error en la recomendación")

# Endpoint para recomendaciones
@app.get("/recomendacion/")
def obtener_recomendacion(title: str):
    try:
        recomendaciones = recomendacion(title)
        return {"recomendaciones": recomendaciones}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error en /recomendacion/: {e}")
        raise HTTPException(status_code=500, detail="Error en la recomendación")
