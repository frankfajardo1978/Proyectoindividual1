# main.py
from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el dataset
df = pd.read_csv("archivoprueba.csv")  # Ruta de tu archivo en Render

app = FastAPI()

# Endpoint de verificación de datos
@app.get("/explore")
async def explore_data():
    return {
        "columns": df.columns.tolist(),
        "num_rows": len(df)
    }

# Función para el sistema de recomendación
def recomendacion(titulo: str) -> List[str]:
    if titulo not in df['titulo'].values:
        raise HTTPException(status_code=404, detail="Título no encontrado")
    
    # Vectorización de los títulos de las películas
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["titulo"])

    # Similaridad de coseno entre la película solicitada y las demás
    idx = df[df['titulo'] == titulo].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Obtener los índices de las 5 películas más similares (excluyendo la misma)
    similar_indices = cosine_similarities.argsort()[-6:-1][::-1]
    recommendations = df.iloc[similar_indices]["titulo"].tolist()
    
    return recommendations

# Endpoint para el sistema de recomendación
@app.get("/recomendacion/{titulo}")
async def get_recommendation(titulo: str):
    try:
        return {"recommendations": recomendacion(titulo)}
    except HTTPException as e:
        raise e

# Ejecutar el servidor en Render
#if __name__ == "__main__":
  #  import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
