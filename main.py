from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Cargar el archivo CSV
data_path = "archivoprueba.csv"
df = pd.read_csv(data_path)

app = FastAPI(title="API de Análisis Exploratorio y Sistema de Recomendación")

# Clase para los títulos de recomendación
class RecomendacionRequest(BaseModel):
    titulo: str

# Ruta de Exploratory Data Analysis (EDA)
@app.get("/eda/wordcloud")
def eda_wordcloud():
    # Generar la nube de palabras para los títulos
    text = " ".join(df["title"].astype(str))  # Asegurarse que 'title' existe en el dataset
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    
    # Guardar la imagen de la nube de palabras
    wordcloud.to_file("wordcloud.png")
    
    return JSONResponse(content={"message": "Nube de palabras generada y guardada como 'wordcloud.png'"})

# Sistema de recomendación
@app.post("/recomendacion")
def recomendacion(request: RecomendacionRequest):
    titulo = request.titulo.lower()
    
    # Verificar que el título existe en el dataset
    if titulo not in df["title"].str.lower().values:
        raise HTTPException(status_code=404, detail="La película no se encontró en la base de datos.")
    
    # Crear una matriz de similitud de texto (bag of words) basada en los títulos
    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(df["title"].astype(str))
    
    # Calcular la similitud de coseno entre las películas
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Encontrar el índice de la película solicitada
    idx = df[df["title"].str.lower() == titulo].index[0]
    
    # Ordenar las películas según similitud y seleccionar las 5 más similares
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_5_indices = [i[0] for i in similarity_scores[1:6]]  # Excluir la misma película
    
    # Obtener los títulos de las 5 películas más similares
    recommended_titles = df.iloc[top_5_indices]["title"].tolist()
    
    return JSONResponse(content={"recomendaciones": recommended_titles})

# Ejemplo de una gráfica de exploración de relaciones
@app.get("/eda/plots")
def eda_plots():
    # Gráfica simple de ejemplo
    plt.figure(figsize=(10, 6))
    df["some_column"].hist(bins=30)  # Reemplazar 'some_column' con una columna de interés
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de valores de some_column")
    plt.savefig("distribution_plot.png")
    
    return JSONResponse(content={"message": "Gráfica de distribución generada y guardada como 'distribution_plot.png'"})

# Iniciar la API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
