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

@app.route('/correlation-heatmap', methods=['GET'])
def correlation_heatmap():
    try:
        # Generar la matriz de correlación
        correlation_matrix = df.corr()
        
        # Crear el gráfico del mapa de calor
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, fmt=".2f")
        
        # Guardar la imagen en un objeto de bytes
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        # Devolver la imagen
        return send_file(img, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)})


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
