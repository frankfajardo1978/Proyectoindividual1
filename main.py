from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
from datetime import datetime
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el archivo CSV
data = pd.read_csv("archivov4.csv", index_col=0)

# Convertir títulos a minúsculas para comparación insensible a mayúsculas
data['title_lower'] = data['title'].str.lower()

# Crear directorio para guardar gráficos
# os.makedirs("graphs", exist_ok=True)

# Crear aplicación FastAPI
app = FastAPI()

@app.get("/")
async def bienvenida():
    return "Bienvenidos al proyecto nr 1 de Francisco Fajardo (Soy Henry)"

# Endpoint: cantidad_filmaciones_mes
@app.get("/mes")
def cantidad_filmaciones_mes(mes: str):
    mes = mes.lower()
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    if mes not in meses:
        return {"error": "Mes ingresado no válido. Por favor, ingrese un mes en español."}
    numero_mes = meses[mes]
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    peliculas_mes = data[data['release_date'].dt.month == numero_mes]
    cantidad = len(peliculas_mes)
    return {"mensaje": f"{cantidad} películas fueron estrenadas en el mes de {mes.capitalize()}"}

# Endpoint: cantidad_filmaciones_dia
@app.get("/dia")
def cantidad_filmaciones_dia(dia: str):
    dia = dia.capitalize()
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    dias_traduccion = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    data['dia_semana'] = data['release_date'].dt.day_name().map(dias_traduccion)
    cantidad = data[data['dia_semana'] == dia].shape[0]
    return {"mensaje": f"{cantidad} películas fueron estrenadas en los días {dia}"}

# Endpoint: score_titulo
@app.get("/titulo")
def score_titulo(titulo: str):
    film = data[data['title'].str.lower() == titulo.lower()]
    if film.empty:
        return {"error": "Película no encontrada"}
    titulo = film.iloc[0]['title']
    año = int(film.iloc[0]['release_year'])
    score = film.iloc[0]['popularity']
    return {"mensaje": f"La película {titulo} fue estrenada en el año {año} con un score/popularidad de {score}"}

# Endpoint: votos_titulo
@app.get("/votos_titulo/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    pelicula = data[data['title'].str.lower() == titulo_de_la_filmacion.lower()]
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

# Endpoint: get_actor
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

# Endpoint: get_director
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


# Sistema de recomendación basado en similitud de coseno
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'].fillna(""))

# Endpoint: recommendation
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
