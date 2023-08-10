from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import re
import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

### Cargando los Datos
df_movies = pd.read_csv('DataSets/df_movies_JOIN.csv', sep=',')

### Eliminando Valores Nulos de los Datos Cualitativos  utilizar en la Recomendación.
ColumnasNaNtoBlancos = ['Franquicia', 'Generos', 'Productores', 'Paises', 'IdiomasH', 'Director']
# Reemplazar los valores nulos en las columnas con un espacio en blanco.
df_movies[ColumnasNaNtoBlancos] = df_movies[ColumnasNaNtoBlancos].fillna(' ').astype(str)

# http://127.0.0.1:8000

###  INICIANDO LAS PRUEBAS CON FASTAPI

app = FastAPI()


### FUNCION Nro. 1
@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    try:
        idioma = idioma.lower()
        mask = df_movies[df_movies['original_language'].str.lower() == idioma]
        
        if mask.empty:
            {'Error': 'Idioma no encontrado. Intente de nuevo...!!! '}    # Si la mascara esta vacía, entonces no encontro ningun idioma.
        
        respuesta = mask.shape[0]   # Si la mascara no esta vacís, entonces registra la cantidad de registros o filas = Total de Películas.         

        return {'idioma':idioma, 'cantidad':respuesta}   # Retorna el resultado

    except Exception as e:
        return {'error': f'Ocurrió un error: {e}'}


#Pelicula = 'Jumanji'
### FUNCION Nro. 2
@app.get('/peliculas_duracion/{pelicula}')   
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    try:
        pelicula = pelicula.lower()
        datos_pelicula = df_movies[df_movies['title'].str.lower() == pelicula]  # Si no lo encuentra se convierte en un Dataframe vacío
        
        if datos_pelicula.empty:
            return {'error': 'Película no encontrada en el DataFrame.'}
        
        duracion = int(datos_pelicula['runtime'].iloc[0])   # Extrae la primera fila de la columna 'runtime' = Duración
        anio = int(datos_pelicula['Anio'].iloc[0])          # Extrae la primera fila de la columna 'Anio' = Anio

        return {'pelicula':pelicula, 'duracion':duracion, 'anio':anio}

    except Exception as e:
        return {'Error': f'Por favor, corrija este error: {e}'}


### FUNCION Nro. 3
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    try:
        franquicia = franquicia.lower()
        datos_franquicia = df_movies[df_movies['Franquicia'].str.lower() == franquicia]
        
        if datos_franquicia.empty:
            return {'error': 'Franquicia no encontrada en el DataFrame.'}
        
        cantidad = int(datos_franquicia['Franquicia'].shape[0])
        ganancia_total = float(datos_franquicia['revenue'].sum())
        
        if cantidad == 0:
            ganancia_promedio = 0
        else:
            ganancia_promedio = float(ganancia_total/cantidad)
        
        return {'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}
        
    except Exception as e:
        return {'Error': f'Por favor, corrija este error: {e}'}


### FUNCION Nro. 4
@app.get('/peliculas_pais/{pais}')    
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    try:
        pais = pais.lower()
        datos_pais = df_movies[df_movies['Paises'].str.lower() == pais]
        
        if datos_pais.empty:
            return {'error': 'País no encontrado en el DataFrame.'}
        
        cantidad = int(datos_pais['Paises'].shape[0]) 
        
        return {'pais':pais, 'cantidad':cantidad}  
  
    except Exception as e:
        return {'Error': f'Por favor, corrija este error: {e}'} 
    

### productora = 'Touchstone Pictures'
### FUNCION Nro. 5
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revenue total y la cantidad de peliculas que realizo '''
    try:
        productora = productora.lower() 
        datos_productora = df_movies[df_movies['Productores'].str.lower().str.contains(productora)]
        
        if datos_productora.empty:
            return {'error': 'Productota no encontrada en el DataFrame.'}
                  
        revenue_total = float(datos_productora['revenue'].sum())
        cantidad = int(datos_productora['revenue'].shape[0])
        
        return {'productora':productora, 'revenue_total': revenue_total,'cantidad':cantidad}
    
    except Exception as e:
        return {'Error': f'Por favor, corrija este error: {e}'}


### NombreDelDirector = 'John Lasseter'
### FUNCION Nro. 6
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    try:
        nombre_director = nombre_director.lower()
        
        datos_director = df_movies[df_movies['Director'].str.lower().str.contains(nombre_director)]    # #Hacemos una lista de ocurrencia de Directores en un DF temporal
      
        if datos_director.empty:
            return {'error': 'Director no encontrado en el DataFrame.'}
        
        # Peliculas del Director con sus respectivas variables       
        Peliculas_del_Director = datos_director[['title', 'Anio', 'revenue', 'budget']]
        
        # Retorno del exito 
        retorno_total_director = float(Peliculas_del_Director['revenue'].sum() / Peliculas_del_Director['budget'].sum())
        
        # Convertir el DataFrame en una lista de diccionarios
        Lista_De_Dicc = Peliculas_del_Director.to_dict('records')
        
        return {'director':nombre_director, 'retorno_total_director':retorno_total_director, 'peliculas':Lista_De_Dicc}
        
    except Exception as e:
        return {'Error': f'Por favor, corrija este error: {e}'}
  

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    try:
        # Convertir el título ingresado por el usuario a minúsculas
        titulo = titulo.lower()
        # Obtener el id de la película que le gustó al usuario
        movie_id = df_movies.loc[df_movies['title'].str.lower()  == titulo, 'id'].iloc[0]

        # Obteneiendo las características de las siguientes variables predictoras del Dataset
        genre_features = df_movies['Generos']
        director_features = df_movies['Director']
        protagonist_features = df_movies['Protagonista']
        actor1_features = df_movies['Actor1']
        actor2_features = df_movies['Actor2']
        popularity_features = df_movies['popularity'].fillna(0).astype(str)  # Convertir a cadena y llenar valores NaN con '0'
        anio_features = df_movies['Anio'].fillna(0).astype(str)  # Convertir a cadena y llenar valores NaN con '0'
        revenue_features = df_movies['revenue'].fillna(0).astype(str)  # Convertir a cadena y llenar valores NaN con '0'
        vote_average_features = df_movies['vote_average'].fillna(0).astype(str)  # Convertir a cadena y llenar valores NaN con '0'
        vote_count_features = df_movies['vote_count'].fillna(0).astype(str)  # Convertir a cadena y llenar valores NaN con '0'
        franquicia_features = df_movies['Franquicia']
        
        # Concatenar todas las características
        all_features = franquicia_features + ' ' + genre_features + ' ' + director_features + ' ' + protagonist_features + ' ' + actor1_features + ' ' + actor2_features + ' ' + popularity_features + ' ' + anio_features + ' ' + revenue_features + ' ' + vote_average_features + ' ' + vote_count_features

        # Crear un objeto CountVectorizer para convertir las características en vectores
        vectorizer = CountVectorizer(analyzer='word', lowercase=True, token_pattern=r'\w+')

        # Obtener la matriz de documentos término-frecuencia (DTM) a partir de las características
        all_features_matrix = vectorizer.fit_transform(all_features)

        # Obtener las características de la película que le gustó al usuario
        movie_features = df_movies.loc[df_movies['id'] == movie_id, 'Franquicia'].iloc[0] + ' ' + df_movies.loc[df_movies['id'] == movie_id, 'Generos'].iloc[0] + ' ' + df_movies.loc[df_movies['id'] == movie_id, 'Director'].iloc[0] + ' ' + df_movies.loc[df_movies['id'] == movie_id, 'Protagonista'].iloc[0] + ' ' + df_movies.loc[df_movies['id'] == movie_id, 'Actor1'].iloc[0] + ' ' + df_movies.loc[df_movies['id'] == movie_id, 'Actor2'].iloc[0] + ' ' + popularity_features.iloc[0] + ' ' + anio_features.iloc[0] + ' ' + revenue_features.iloc[0] + ' ' + vote_average_features.iloc[0] + ' ' + vote_count_features.iloc[0]
        movie_features_matrix = vectorizer.transform([movie_features])

        # Calcular la similitud del coseno entre la película que le gustó al usuario y todas las demás películas
        similarities = cosine_similarity(movie_features_matrix, all_features_matrix)

        # Obtener los índices de las películas más similares
        similar_indices = similarities.argsort()[0][-6:-1]

        # Obtener los títulos de las películas más similares para la recomendación
        recomendacion = df_movies.loc[similar_indices, 'title']
        
        recomendacion = recomendacion.tolist()

        return {'Películas Recomendadas': recomendacion}
    
    except IndexError:
        return {'Error': 'Película no encontrada. Intente de nuevo...!!! '}
    
    except Exception as e:
        return {'error': f'Ocurrió un error: {e}'}
