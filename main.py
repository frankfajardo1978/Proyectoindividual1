import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Cargar el dataset y verificar la existencia de la columna 'title'
try:
    df = pd.read_csv("archivov4.csv")
    if 'title' not in df.columns:
        raise ValueError("La columna 'title' no se encuentra en el dataset")
except FileNotFoundError:
    raise FileNotFoundError("El archivo 'archivov4.csv' no se encuentra en el directorio actual")
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    raise

app = FastAPI()

# Modelo para manejar los datos entrantes
class Item(BaseModel):
    title: str
    rating: Optional[float] = None

# Ruta de prueba para verificar que el servidor está en funcionamiento
@app.get("/")
def read_root():
    return {"message": "API activa y funcionando correctamente"}

# Ruta de ejemplo que utiliza la columna 'title'
@app.get("/search/")
def search_title(title: str):
    try:
        # Validación de entrada
        if not title:
            raise HTTPException(status_code=400, detail="El parámetro 'title' no puede estar vacío")

        # Búsqueda en el dataframe
        result = df[df['title'].str.contains(title, case=False, na=False)]
        if result.empty:
            return {"message": "No se encontraron coincidencias"}

        return result.to_dict(orient="records")
    except Exception as e:
        print(f"Error en /search/: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
