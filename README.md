# PROYECTO INDIVIDUAL NRO 1

Este proyecto consiste en la creación de un sistema de consultas y recomendaciones de películas, utilizando técnicas de ETL (Extracción, Transformación y Carga) y de análisis exploratorio de datos (EDA) para la preparación de la información. Además, se desarrolló una API que permite al usuario acceder a diversas funcionalidades para obtener datos específicos sobre películas y recomendaciones personalizadas.

##1-ETL

En el proceso de ETL https://github.com/frankfajardo1978/Proyectoindividual1/blob/main/ETL.ipynb realizamos las siguientes transformaciones a los archivos moviesdataset.csv y credits.csv:

###ETL Archivo movies_dataset.csv.
- Observacion de dimensiones del archivo.
- Validacion de variables catergoricas y numericas
- Desanidacion de columnas
- Relleno de valores nulos con numeros ceros "0" y eliminacion de valores nulos en columnas de fechas.
- Eliminacion de columnas inecesarias.
- Formato de columnas de fechas.
- Creacion de columnas de año de lanzamiento.

###ETL Archivo movies_dataset.csv.
- Observacion de dimensiones del archivo.
- Desanidacion de la columna cast
- Separacion de columnas y extraccion de actores.
- Desanidacion y separacion de directores de la columna crew
- Union de columnas desanidadas.

Por ultimo se unifican los dataset de credits y movies_dataset para tener el dataset definitivo con el cual se trabajara en la API.


##2-DESCRIPCION DE LAS APIS.

Basado en el archivo creado y depurado de Movies_dataset y Credits se genero el  sistemas de consultas y recomendaciones contenidos en nuestra API contenida en main.py https://github.com/frankfajardo1978/Proyectoindividual1/blob/main/main.py,  a continuacion una breve descripcion de cada una.


### Cantidad de filmaciones por mes.
Esta es una funcion en la cual se ingresa el mes deseado y devuelve la cantidad depeliculas estrenadas para dicho mes.

### Cantidad  de filmaciones por día.
Esta es una funcion en la cual se ingresa un dia de la semana y devuelve la cantidad total de peliculas estrenadas en el dia especificado.

### Score por título.
Este endpoint consiste en ingresar el nombre de una pelicula y devolver el nombre con el año de y estreno la popularidad de la misma.

### Votos_por título.
Este endpoint consiste en realizar una consulta por titulo de pelicula y devolver titulo de la pelicula, la votacion y el promedio de votacion.

### Actor.
Esta funcion ermite la consulta del nombre de un actor y devuelve el mismo con retorno total y el promedio de retorno.

### Director.
Esta funcion permite consultar el nombre de un director, retornando el mismo,  retorno total y peliculas dirigidas.

### Sistema de recomendación basado en similitud de coseno.
Esta es una funcion que permite realizar una consulta de un nombre de pelicula y el sistema devolvera 5 recomendaciones de titulos similares basados en la columna "overview" del dataset.

###3-EDA
El notebook del EDA se locliza en https://github.com/frankfajardo1978/Proyectoindividual1/blob/main/EDA.ipynb.
En este punto en particular no nos enfocaos en detectar valores nulos, duplicados, faltantes y validacion de tipos de datos se realizaon en el ETL, por tanto aqui nos enfocamos en relizar lo siguiente:

- Mapa de correlacion basado en Pearson con cuadro de calor
- Mapa de distribucion.
- Visualizacion de Outliers
- Nuebs de Palabras
- 


### 4-Deployment

En este punto se podran vaiidar las funciones API en funcionamiento en el link https://proyectoindividual1-tvr2.onrender.com.
