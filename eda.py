import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Cargar los datos
df = pd.read_csv('archivov4.csv')

# Información general del DataFrame
print(df.info())
print(df.describe())

# Generar una nube de palabras para los títulos
titles_text = ' '.join(df['title'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)

# Mostrar la nube de palabras
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
