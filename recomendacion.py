# recomendacion.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar los datos y preparar el modelo dentro de la función para evitar cargar en cada instancia
def cargar_modelo():
    df = pd.read_csv('archivoprueba.csv')
    df['title'] = df['title'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['title'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return df, cosine_sim

def recomendacion(titulo):
    df, cosine_sim = cargar_modelo()
    idx = df[df['title'].str.lower() == titulo.lower()].index
    if len(idx) == 0:
        return []

    sim_scores = list(enumerate(cosine_sim[idx[0]]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:6]]
    return df['title'].iloc[sim_indices].tolist()
