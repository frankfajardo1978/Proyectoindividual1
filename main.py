from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar y preparar datos
data = pd.read_csv("archivov4.csv")
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['title_lower'] = data['title'].str.lower()
data['overview'] = data['overview'].fillna('')
data['genres'] = data['genres'].fillna('')
data['content'] = data['overview'] + " " + data['genres']
data = data[['title', 'title_lower', 'release_date', 'overview', 'genres', 'content', 'vote_count', 'vote_average', 'cast_names', 'director', 'return', 'release_year', 'budget']].dropna(subset=['title'])

# Crear directorio para gráficos
os.makedirs("graphs", exist_ok=True)

# Instancia de la aplicación FastAPI
app = FastAPI()

# Crear la matriz TF-IDF para recomendaciones
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['content'])

# Diccionarios auxiliares
meses = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}
dias_traduccion = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
}
data['dia_semana'] = data['release_date'].dt.day_name().map(dias_traduccion)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de información de películas"}

# Función para generar nube de palabras
def generate_wordcloud():
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data['title']))
    path = "graphs/wordcloud.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(path)
    plt.close()
    return path

# Función para generar histograma de longitud de resúmenes
def generate_histogram():
    data['overview_length'] = data['overview'].apply(lambda x: len(str(x).split()))
    path = "graphs/histogram.png"
    plt.figure(figsize=(10, 5))
    sns.histplot(data['overview_length'], bins=30, kde=True)
    plt.title("Distribución de la longitud de los resúmenes")
    plt.xlabel("Número de palabras")
    plt.ylabel("Frecuencia")
    plt.savefig(path)
    plt.close()
    return path

@app.get("/mes")
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    if mes not in meses:
        return {"error": "Mes ingresado no válido. Por favor, ingrese un mes en español."}
    numero_mes = meses[mes]
    peliculas_mes = data[data['release_date'].dt.month == numero_mes]
    return {"mensaje": f"{len(peliculas_mes)} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}"}

@app.get("/dia")
def cantidad_filmaciones_dia(dia: str):
    dia = dia.capitalize()
    cantidad = data[data['dia_semana'] == dia].shape[0]
    return {"mensaje": f"{cantidad} películas fueron estrenadas en los días {dia}"}

# Ensure all titles are lowercase for matching purposes
data['title_lower'] = data['title'].str.lower()

# Asegúrate de que todas las cadenas en 'title' estén en minúsculas
data['title_lower'] = data['title'].str.lower()

@app.get("/titulo")
def score_titulo(titulo: str):
    # Filtra las películas con el título buscado en minúsculas
    film = data[data['title_lower'] == titulo.lower()]
    
    # Verifica si la película existe y si tiene datos completos
    if film.empty or pd.isnull(film.iloc[0]['release_year']) or pd.isnull(film.iloc[0]['popularity']):
        return {"error": "Película no encontrada o datos incompletos"}
    
    titulo = film.iloc[0]['title']
    año = int(film.iloc[0]['release_year'])
    score = film.iloc[0]['popularity']
    
    return {"mensaje": f"La película {titulo} fue estrenada en el año {año} con un score/popularidad de {score}"}



@app.get("/votos_titulo/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    pelicula = data[data['title_lower'] == titulo_de_la_filmacion.lower()]
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    votos = pelicula.iloc[0]['vote_count']
    promedio_votos = pelicula.iloc[0]['vote_average']
    titulo = pelicula.iloc[0]['title']
    anio = int(pelicula.iloc[0]['release_year'])
    if votos < 2000:
        return {"message": f"La película '{titulo}' no cumple con el requisito de 2000 valoraciones."}
    return {
        "message": f"La película '{titulo}' fue estrenada en el año {anio}.",
        "votos_totales": votos,
        "promedio_votos": promedio_votos
    }

@app.get("/actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actor_data = data[data['cast_names'].str.contains(nombre_actor, case=False, na=False)]
    if actor_data.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")
    cantidad_peliculas = actor_data.shape[0]
    retorno_total = actor_data['return'].sum()
    promedio_retorno = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0
    return {
        "mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} filmaciones, "
                   f"el mismo ha conseguido un retorno de {retorno_total} con un promedio de {promedio_retorno} por filmación"
    }

@app.get("/director/{nombre_director}")
def get_director(nombre_director: str):
    director_data = data[data['director'].str.contains(nombre_director, case=False, na=False)]
    if director_data.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")
    retorno_total = director_data['return'].sum()
    peliculas_info = [
        {
            "nombre_pelicula": row['title'],
            "fecha_lanzamiento": row['release_year'],
            "retorno_individual": row['return'],
            "costo": row['budget'],
            "ganancia": row['budget'] * row['return']
        }
        for _, row in director_data.iterrows()
    ]
    return {
        "nombre_director": nombre_director,
        "retorno_total": retorno_total,
        "peliculas": peliculas_info
    }

@app.get("/wordcloud/")
async def wordcloud():
    path = generate_wordcloud()
    return FileResponse(path, media_type="image/png")

@app.get("/histogram/")
async def histogram():
    path = generate_histogram()
    return FileResponse(path, media_type="image/png")

@app.get("/recommendation/")
async def recommendation(titulo: str):
    titulo = titulo.lower()
    if titulo not in data['title_lower'].values:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    idx = data.index[data['title_lower'] == titulo].tolist()[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    recommendations = data.iloc[similar_indices]['title'].tolist()
    return {"recommendations": recommendations}
