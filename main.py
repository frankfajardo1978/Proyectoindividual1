import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

# Cargar datos y eliminar columna 'Unnamed' si está presente
data = pd.read_csv("archivov4.csv", index_col=0)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]  # Eliminar columnas con 'Unnamed'
data = data[['title', 'overview']].dropna()  # Mantener solo columnas de título y resumen, eliminando nulos

# Crear directorio para guardar gráficos si no existe
os.makedirs("graphs", exist_ok=True)

# Generación del mapa de calor de la matriz de correlaciones para variables numéricas
def generate_correlation_heatmap():
    # Seleccionar solo las columnas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    
    # Verificar si existen columnas numéricas
    if numeric_data.empty:
        raise ValueError("No hay columnas numéricas en el dataset para generar un mapa de calor de correlación.")
    
    # Calcular matriz de correlaciones
    correlation_matrix = numeric_data.corr()

    # Crear el mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Mapa de calor de la matriz de correlaciones para variables numéricas")
    path = "graphs/correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    return path

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

# Endpoint para mostrar el mapa de calor de la matriz de correlaciones
@app.get("/correlation-heatmap/")
async def correlation_heatmap():
    try:
        path = generate_correlation_heatmap()
        return FileResponse(path, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

# Función de recomendación optimizada sin 'title_lower'
@app.get("/recommendation/")
async def recommendation(titulo: str):
    titulo = titulo.lower()
    if titulo not in data['title'].str.lower().values:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    # Obtener el índice de la película
    idx = data.index[data['title'].str.lower() == titulo].tolist()[0]
    
    # Calcular similitud de coseno
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Obtener las películas más similares (excluyendo la misma)
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    recommendations = data.iloc[similar_indices]['title'].tolist()
    
    return {"recommendations": recommendations}

# Condicional para ejecutar el servidor si se ejecuta directamente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
