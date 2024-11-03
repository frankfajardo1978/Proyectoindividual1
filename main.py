import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException

# Cargar datos
data = pd.read_csv("/mnt/data/archivov4.csv")
data = data[['title', 'overview']].dropna()  # Mantener solo las columnas de título y resumen, eliminando nulos

# Convertir títulos a minúsculas para comparación insensible a mayúsculas
data['title_lower'] = data['title'].str.lower()

# Exploración de datos
def exploratory_data_analysis(data):
    # Nube de palabras para títulos
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data['title']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras de títulos")
    plt.show()

    # Análisis de longitud de los resúmenes
    data['overview_length'] = data['overview'].apply(lambda x: len(str(x).split()))
    sns.histplot(data['overview_length'], bins=30, kde=True)
    plt.title("Distribución de la longitud de los resúmenes")
    plt.xlabel("Número de palabras")
    plt.ylabel("Frecuencia")
    plt.show()

    # Identificación de outliers en la longitud de los resúmenes
    overview_q1 = data['overview_length'].quantile(0.25)
    overview_q3 = data['overview_length'].quantile(0.75)
    iqr = overview_q3 - overview_q1
    outliers = data[(data['overview_length'] < (overview_q1 - 1.5 * iqr)) | 
                    (data['overview_length'] > (overview_q3 + 1.5 * iqr))]
    print("Posibles outliers en longitud de resúmenes:")
    print(outliers[['title', 'overview_length']])

# Ejecutar el análisis exploratorio
exploratory_data_analysis(data)

# Optimización de recomendación con pre-cálculo de TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'].fillna(""))

app = FastAPI()

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
