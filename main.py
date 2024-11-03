from fastapi import FastAPI
import pandas as pd
from datetime import datetime
from fastapi import FastAPI








# Cargar el archivo CSV al iniciar la aplicación
data = pd.read_csv("archivov4.csv")


app = FastAPI()

@app.get("/")
def read_root():
    return {"Bienvenido"}     

@app.get("/mes")
def cantidad_filmaciones_mes(mes: str):
    # Convertir el mes a minúsculas para evitar problemas de mayúsculas
    mes = mes.lower()
    
    # Diccionario para convertir el nombre del mes en español al número correspondiente
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    
    # Validar si el mes ingresado es válido
    if mes not in meses:
        return {"error": "Mes ingresado no válido. Por favor, ingrese un mes en español."}
    
    # Obtener el número del mes correspondiente
    numero_mes = meses[mes]
    
    # Convertir la columna de fechas al tipo datetime si es necesario
    data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
    
    # Filtrar las películas estrenadas en el mes especificado
    peliculas_mes = data[data['release_date'].dt.month == numero_mes]
    cantidad = len(peliculas_mes)
    
    return {"mensaje": f"{cantidad} cantidad de películas fueron estrenadas en el mes de {mes.capitalize()}"}




# Definir el endpoint para cantidad_filmaciones_dia
@app.get("/dia")
def cantidad_filmaciones_dia(dia: str):
    dia = dia.capitalize()  # Ajustar la capitalización para comparación
    cantidad = data[data['dia_semana'] == dia].shape[0]  # Contar películas
    return {"mensaje": f"{cantidad} películas fueron estrenadas en los días {dia}"}


# Diccionario para traducir nombres de días de inglés a español
dias_traduccion = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo"
}

# Convertir la columna de fecha a formato de fecha y extraer el día de la semana en inglés
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['dia_semana'] = data['release_date'].dt.day_name()
data['dia_semana'] = data['dia_semana'].map(dias_traduccion)  # Traducir a español






@app.get("/titulo")
def score_titulo(titulo: str):
    # Buscar la película por título
    film = data[data['title'].str.lower() == titulo.lower()]
    
    # Verificar si se encontró la película
    if film.empty:
        return {"error": "Película no encontrada"}
    
    # Extraer la información deseada
    titulo = film.iloc[0]['title']
    año = int(film.iloc[0]['release_year'])
    score = film.iloc[0]['popularity']
    
    return {
        "mensaje": f"La película {titulo} fue estrenada en el año {año} con un score/popularidad de {score}"
    }


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de información de películas"}

@app.get("/votos_titulo/{titulo_de_la_filmacion}")
def votos_titulo(titulo_de_la_filmacion: str):
    # Filtrar la película por título
    pelicula = data[data['title'].str.lower() == titulo_de_la_filmacion.lower()]
    
    # Comprobar si existe la película
    if pelicula.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Obtener el número de votos y el promedio de votos
    votos = pelicula.iloc[0]['vote_count']
    promedio_votos = pelicula.iloc[0]['vote_average']
    titulo = pelicula.iloc[0]['title']
    anio = int(pelicula.iloc[0]['release_year'])

    # Validar si cumple con al menos 2000 votos
    if votos < 2000:
        return {"message": f"La película '{titulo}' no cumple con el requisito de 2000 valoraciones."}
    
    # Retornar la información si cumple el requisito
    return {
        "message": f"La película '{titulo}' fue estrenada en el año {anio}.",
        "votos_totales": votos,
        "promedio_votos": promedio_votos
    }


@app.get("/actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    try:
        # Filtrar las películas en las que ha participado el actor
        actor_data = data[data['cast_names'].str.contains(nombre_actor, case=False, na=False)]
        
        # Verificar si el actor tiene registros en el dataset
        if actor_data.empty:
            raise HTTPException(status_code=404, detail="Actor no encontrado")
        
        # Calcular la cantidad de películas, retorno total y promedio de retorno
        cantidad_peliculas = actor_data.shape[0]
        retorno_total = actor_data['return'].sum()
        promedio_retorno = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0

        # Responder con el mensaje formateado
        return {
            "mensaje": f"El actor {nombre_actor} ha participado de {cantidad_peliculas} filmaciones, "
                       f"el mismo ha conseguido un retorno de {retorno_total} con un promedio de {promedio_retorno} por filmación"
        }
    
    except Exception as e:
        # Manejo de errores para dar más detalles en el mensaje de error
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/director/{nombre_director}")
def get_director(nombre_director: str):
    try:
        # Filtrar las películas dirigidas por el director
        director_data = data[data['director'].str.contains(nombre_director, case=False, na=False)]
        
        # Verificar si el director tiene registros en el dataset
        if director_data.empty:
            raise HTTPException(status_code=404, detail="Director no encontrado")
        
        # Calcular el retorno total del director
        retorno_total = director_data['return'].sum()

        # Crear una lista de películas con la información solicitada
        peliculas_info = []
        for _, row in director_data.iterrows():
            nombre_pelicula = row['title']
            fecha_lanzamiento = row['release_year']
            retorno_individual = row['return']
            costo = row['budget']
            ganancia = costo * retorno_individual
            
            peliculas_info.append({
                "nombre_pelicula": nombre_pelicula,
                "fecha_lanzamiento": fecha_lanzamiento,
                "retorno_individual": retorno_individual,
                "costo": costo,
                "ganancia": ganancia
            })
        
        # Responder con el retorno total y la información de cada película
        return {
            "mensaje": f"El director {nombre_director} ha conseguido un retorno total de {retorno_total}.",
            "peliculas": peliculas_info
        }
    
    except Exception as e:
        # Manejo de errores para dar más detalles en el mensaje de error
        raise HTTPException(status_code=500, detail=str(e))



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI
from pydantic import BaseModel

# Inicializar FastAPI
app = FastAPI()

# Cargar solo las columnas necesarias
file_path = 'archivov4.csv'  # Cambia la ruta si es necesario
df = pd.read_csv(file_path, usecols=['title', 'overview']).fillna('')

# Crear la matriz TF-IDF en el momento del inicio para minimizar el uso de memoria
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Definir la función de recomendación
def recomendacion(titulo):
    # Obtener el índice de la película que coincide con el título
    idx = df.index[df['title'] == titulo].tolist()
    if not idx:
        return ["Película no encontrada."]
    
    idx = idx[0]

    # Calcular la similitud solo para la película solicitada
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Ordenar y seleccionar las 5 películas más similares
    sim_scores = cosine_sim.argsort()[-6:-1][::-1]
    return df['title'].iloc[sim_scores].tolist()

# Configurar la estructura de entrada para la API
class MovieTitle(BaseModel):
    titulo: str

# Crear el endpoint de la API
@app.post("/recomendacion/")
def get_recomendacion(data: MovieTitle):
    return {"recomendaciones": recomendacion(data.titulo)}
