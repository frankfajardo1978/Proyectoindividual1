from fastapi import FastAPI
from recommendation import recomendacion  # Importar la función de recomendación

app = FastAPI()

@app.get("/recomendar/")
def recomendar(titulo: str):
    return {"recomendaciones": recomendacion(titulo)}
