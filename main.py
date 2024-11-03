from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

app = FastAPI()

# Cargar el dataset
data_path = "/mnt/data/archivov4.csv"  # Ajusta el nombre según corresponda
df = pd.read_csv(data_path)

# Verificar que el dataset contiene la columna 'titulo'
if 'titulo' not in df.columns:
    raise ValueError("La columna 'titulo' no se encuentra en el dataset")

# Generar la nube de palabras
def generar_nube_palabras():
    titles_text = " ".join(df['titulo'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(titles_text)
    wordcloud.to_file("nube_palabras.png")
    return "nube_palabras.png"

# Modelo de recomendación basado en similitud de títulos
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df['titulo'].astype(str))
cosine_sim = cosine_similarity(tfidf_matrix)

class RecomendacionRequest(BaseModel):
    titulo: str

@app.get("/eda/nube_palabras")
async def obtener_nube_palabras():
    # Generar nube de palabras y devolverla
    try:
        file_path = generar_nube_palabras()
        return {"nube_palabras": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recomendacion/")
async def recomendacion(request: RecomendacionRequest):
    titulo = request.titulo
    if titulo not in df['titulo'].values:
        raise HTTPException(status_code=404, detail="La película no fue encontrada")
    
    idx = df[df['titulo'] == titulo].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:6]]
    recomendaciones = df['titulo'].iloc[top_indices].tolist()
    
    return {"recomendaciones": recomendaciones}

# Iniciar el servidor
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
