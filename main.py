import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Cargar el dataset y verificar la existencia de la columna 'title'
try:
    df = pd.read_csv("archivov4.csv")
    if 'title' not in df.columns:
        raise ValueError("La columna 'title' no se encuentra en el dataset")
except FileNotFoundError:
    raise FileNotFoundError("El archivo 'archivov4.csv' no se encuentra en el directorio actual")

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
    result = df[df['title'].str.contains(title, case=False, na=False)]
    if result.empty:
        return {"message": "No se encontraron coincidencias"}
    return result.to_dict(orient="records")
