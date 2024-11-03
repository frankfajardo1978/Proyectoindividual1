from fastapi import FastAPI
from recommender import recomendacion  # Importa la función de recomendación
import pandas as pd

# Inicializar FastAPI
app = FastAPI()

# Endpoint de recomendación
@app.get("/recomendacion/{titulo}")
def get_recomendacion(titulo: str):
    recomendaciones = recomendacion(titulo)
    if "error" in recomendaciones:
        return {"error": recomendaciones["error"]}
    return {"recomendaciones": recomendaciones}
