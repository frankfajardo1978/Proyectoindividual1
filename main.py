import numpy as np

# Load data
data = pd.read_csv("archivov4.csv")
data = data[['title', 'overview']].dropna()  # Keep only title and overview columns, removing null values

# Convert titles to lowercase for case-insensitive comparison
data['title_lower'] = data['title'].str.lower()

# Create directory to save graphs
os.makedirs("graphs", exist_ok=True)

# EDA and graph generation functions
def generate_wordcloud():
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(data['title']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Titles")
    path = "graphs/wordcloud.png"
    plt.savefig(path)
    plt.close()
    return path

def generate_histogram():
    data['overview_length'] = data['overview'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(data['overview_length'], bins=30, kde=True)
    plt.title("Overview Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    path = "graphs/histogram.png"
    plt.savefig(path)
    plt.close()
    return path

def generate_correlation_heatmap():
    # Calculate TF-IDF matrix and compute cosine similarity between items
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'].fillna(""))
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(cosine_sim_matrix, cmap="coolwarm", square=True)
    plt.title("Cosine Similarity Correlation Heatmap")
    path = "graphs/correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    return path

# Optimized recommendation setup with precomputed TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(data['overview'].fillna(""))

app = FastAPI()

# Endpoint for word cloud
@app.get("/wordcloud/")
async def wordcloud():
    path = generate_wordcloud()
    return FileResponse(path, media_type="image/png")

# Endpoint for histogram
@app.get("/histogram/")
async def histogram():
    path = generate_histogram()
    return FileResponse(path, media_type="image/png")

# Endpoint for correlation heatmap
@app.get("/correlation_heatmap/")
async def correlation_heatmap():
    path = generate_correlation_heatmap()
    return FileResponse(path, media_type="image/png")

# Optimized recommendation function
@app.get("/recommendation/")
async def recommendation(titulo: str):
    titulo = titulo.lower()

    if titulo not in data['title_lower'].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Get movie index
    idx = data.index[data['title_lower'] == titulo].tolist()[0]
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    # Get most similar movies (excluding the input movie itself)
    similar_indices = cosine_sim.argsort()[-6:-1][::-1]
    recommendations = data.iloc[similar_indices]['title'].tolist()
    
    return {"recommendations": recommendations}
