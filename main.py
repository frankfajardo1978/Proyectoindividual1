from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load data from CSV
data = pd.read_csv("archivov4.csv")
data = data[['title', 'overview']]  # Filter to only title and overview columns

# Convert titles to lowercase for case-insensitive comparison
data['title_lower'] = data['title'].str.lower()

app = FastAPI()

# @app.get("/")
# async def read_root():
#    return {"message": "Welcome to the Movie Recommendation API"}

# Endpoint to get movie titles and overviews
#@app.get("/movies/")
# async def get_movies():
#    return data[['title', 'overview']].to_dict(orient="records")

# Function for recommending movies based on title similarity
@app.get("/recommendation/")
async def recommendation(titulo: str):
    # Convert input title to lowercase for case-insensitive matching
    titulo = titulo.lower()
    
    # Check if the title exists in the dataset
    if titulo not in data['title_lower'].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    # TF-IDF Vectorizer to convert the overview text into feature vectors
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'].fillna(""))

    # Get index of the input title
    idx = data.index[data['title_lower'] == titulo].tolist()[0]
    
    # Calculate cosine similarity of the movie overview with all others
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get indices of the most similar movies, excluding the movie itself
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    
    # Get movie titles of the top 5 recommendations
    recommendations = data.iloc[similar_indices]['title'].tolist()
    
    return {"recommendations": recommendations}
