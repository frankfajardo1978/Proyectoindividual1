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
data = pd.read_csv("/archivov4.csv")
data = data.dropna()  # Eliminamos filas con valores nulos para simplificar

# Convertir títulos a minúsculas para comparación insensible a mayúsculas
data['title_lower'] = data['title'].str.lower()

# Crear directorio para guardar gráficos
os.makedirs("graphs", exist_ok=True)

# Generación del mapa de calor de la matriz de correlaciones para variables numéricas
def generate_correlation_heatmap():
    # Seleccionar solo las columnas numéricas
    numeric_data = data.select_dtypes(include=['number'])
    
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
    wordcloud = WordCloud(width=800, height=400, background_color
