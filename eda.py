import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('archivov4.csv')  # Reemplaza con la ruta de tu archivo

# Generar la nube de palabras de los títulos de películas
titles_text = " ".join(data['title'].dropna().astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(titles_text)

# Visualizar la nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de Palabras de los Títulos de Películas")
plt.show()
