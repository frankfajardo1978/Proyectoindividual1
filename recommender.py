import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar los datos
data = pd.read_csv('archivov4')  # Reemplaza con la ruta de tu archivo
filtered_data = data[(data['overview'].notna()) & (data['vote_count'] >= 50)].reset_index(drop=True)

# Vectorizar los resúmenes
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filtered_data['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Función de recomendación
def recomendacion(titulo):
    if titulo.lower() not in filtered_data['title'].str.lower().values:
        return {"error": "La película no se encuentra en la base de datos."}
    
    # Obtener el índice y las similitudes
    idx = filtered_data[filtered_data['title'].str.lower() == titulo.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:6]]
    
    # Retornar los títulos
    return filtered_data['title'].iloc[sim_indices].tolist()
