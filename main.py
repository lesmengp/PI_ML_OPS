from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import JSONResponse


### Carga de Datos
df_Funcion_5 = pd.read_csv('DadatesClean/df_Funcion_5.csv')
df_Funcion_6 = pd.read_csv('DadatesClean/df_Funcion_6.csv')
df_movies = pd.read_csv('DadatesClean/df_movies.csv', sep=';')

##   https://pi-ml-ops-srp.onrender.com

###  INICIANDO LAS PRUEBAS CON FASTAPI


app = FastAPI()
class Item(BaseModel):
    cantidad1: float
    revenue_total1: float


### FUNCION Nro. 1
@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    try:
        mask = df_movies[df_movies['original_language'] == idioma]
        respuesta = int(mask['original_language'].count())
    except (ValueError, SyntaxError):
        pass 
    return {'idioma':idioma, 'cantidad':respuesta}


### FUNCION Nro. 2
@app.get('/peliculas_duracion/{pelicula}')   
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    try:
        duracion = df_movies[df_movies['title'] == pelicula]['runtime']
        anio = int(df_movies[df_movies['title'] == pelicula]['Anio'])
    except (ValueError, SyntaxError):
        pass 
    return {'pelicula':pelicula, 'duracion':duracion, 'anio':anio}


### FUNCION Nro. 3
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    try:
        cantidad = int(df_movies[df_movies['Franquicia'] == franquicia]['Franquicia'].count())
        ganancia_total = float(df_movies[df_movies['Franquicia'] == franquicia]['revenue'].sum())
        if cantidad == 0:
            ganancia_promedio = 0
        else:
            ganancia_promedio = float(ganancia_total/cantidad)
    except (ValueError, SyntaxError):
        pass 
    return {'franquicia':franquicia, 'cantidad':cantidad, 'ganancia_total':ganancia_total, 'ganancia_promedio':ganancia_promedio}


### FUNCION Nro. 4    
@app.get('/peliculas_pais/{pais}')
def peliculas_pais(pais:str):
    '''Ingresas el pais, retornando la cantidad de peliculas producidas en el mismo'''
    try:
        respuesta = int(df_movies[df_movies['Paises'] == pais]['id'].count())
    except (ValueError, SyntaxError):
        pass 
    return {'pais':pais, 'cantidad':respuesta}


#productora = 'Touchstone Pictures'
### FUNCION Nro. 5
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revenue total y la cantidad de peliculas que realizo '''
    try:
        revenue_total = float(df_Funcion_5[df_Funcion_5['companies'].str.contains(productora)]['revenue'].sum())
        cantidad = int(df_Funcion_5[df_Funcion_5['companies'].str.contains(productora)]['revenue'].count())
    except (ValueError, SyntaxError):
        pass 
    return {'productora':productora, 'revenue_total': revenue_total,'cantidad':cantidad}


### FUNCION Nro. 6
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    try:
        #Hacemos una lista de ocurrencia de Directores en un DF temporal
        df_Director = df_Funcion_6[df_Funcion_6['Nombre'].str.contains(nombre_director)]
        
        # Peliculas del Director con sus respectivas variables       
        Peliculas_del_Director = df_Director[['title', 'release_year', 'revenue', 'budget']]
        
        # Retorno del exito 
        retorno_total_director = (Peliculas_del_Director['revenue'].sum() / Peliculas_del_Director['budget'].sum())
        
        # Convertir el DataFrame en una lista de diccionarios
        Lista_De_Dicc = Peliculas_del_Director.to_dict('records')
        
        # Convertir la lista de diccionarios a formato JSON
        # Lista_De_Dicc_a_json = json.dumps(Lista_De_Dicc)
        
    except (ValueError, SyntaxError):
        pass 
    return {'director':nombre_director, 'retorno_total_director':retorno_total_director, 'peliculas':Lista_De_Dicc}
  

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas la productora, entregandote el revenue total y la cantidad de peliculas que realizo '''
    try:
        # Obtener el id de la película que le gustó al usuario
        movie_id = df_movies.loc[df_movies['title'] == titulo, 'id'].iloc[0]
        
        # Obteneiendo las características de las siguientes variables predictoras del Dataset
        genre_features = df_movies['Generos']
        director_features = df_movies['Director']
        protagonist_features = df_movies['Protagonista']
        actor1_features = df_movies['Actor1']
        actor2_features = df_movies['Actor2']
        anio_features = df_movies['Anio'].fillna(0).astype(str)  # Convertir a cadena y llenar valores NaN con '0'
        
        # Concatenar todas las características
        all_features = genre_features + ' ' + director_features + ' ' + protagonist_features + ' ' + actor1_features + ' ' + actor2_features + ' ' + revenue_features

        # Crear un objeto CountVectorizer para convertir las características en vectores
        vectorizer = CountVectorizer(analyzer='word', lowercase=True, token_pattern=r'\w+')

        # Obtener la matriz de documentos término-frecuencia (DTM) a partir de las características
        all_features_matrix = vectorizer.fit_transform(all_features)

        # Obtener las características de la película que le gustó al usuario
        movie_features = df_movies.loc[df_movies['id'] == movie_id, 'Generos'].iloc[0] + ' ' + df_sr.loc[df_sr['id'] == movie_id, 'Director'].iloc[0] + ' ' + df_sr.loc[df_sr['id'] == movie_id, 'Protagonista'].iloc[0] + ' ' + df_sr.loc[df_sr['id'] == movie_id, 'Actor1'].iloc[0] + ' ' + df_sr.loc[df_sr['id'] == movie_id, 'Actor2'].iloc[0] + ' ' + anio_features.iloc[0]
        movie_features_matrix = vectorizer.transform([movie_features])

        # Calcular la similitud del coseno entre la película que le gustó al usuario y todas las demás películas
        similarities = cosine_similarity(movie_features_matrix, all_features_matrix)

        # Obtener los índices de las películas más similares
        similar_indices = similarities.argsort()[0][-6:-1]

        # Obtener los títulos de las películas más similares
        similar_movies = df_movies.loc[similar_indices, 'title'] 
        
    except (ValueError, SyntaxError):
        pass 
    return {'lista recomendada': similar_movies}