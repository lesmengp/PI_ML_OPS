from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional   # El Tipo del atributo podria ser opcional

from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse

import pandas as pd
import re
import webbrowser
import textwrap

import traceback
import logging

import json

app = FastAPI()
#templates = Jinja2Templates(directory="templates")

#df_movies = pd.read_csv('DataSets/movies_dataset_RI.csv', sep=';')
#df_movies = pd.read_csv('DataSets/movies_dataset_RI.csv', sep=';', encoding='utf-8')
#df = pd.read_csv('DataSets/movies_dataset_RI.csv', sep=';', encoding='latin-1') 

df = pd.read_csv('../DataSetPI-Cleaning/df_merged-OK.csv', sep=';')

# http://127.0.0.1:8000

###  INICIANDO LAS PRUEBAS CON FASTAPI


### FUNCION Nro. 1
@app.get('/peliculas_idioma/{idioma}')
def peliculas_idioma(idioma:str):
    '''Ingresas el idioma, retornando la cantidad de peliculas producidas en el mismo'''
    try:
        mask = df[df['original_language'] == idioma]
        respuesta = int(mask['original_language'].count())
    except (ValueError, SyntaxError):
        pass 
    return {'idioma':idioma, 'cantidad':respuesta}


### FUNCION Nro. 2
@app.get('/peliculas_duracion/{pelicula}')   
def peliculas_duracion(pelicula:str):
    '''Ingresas la pelicula, retornando la duracion y el año'''
    try:
        respuesta = df[df['title'] == pelicula]['runtime']
        anio = int(df[df['title'] == pelicula]['release_year'])
    except (ValueError, SyntaxError):
        pass 
    return {'pelicula':pelicula, 'duracion':respuesta, 'anio':anio}


### FUNCION Nro. 3
@app.get('/franquicia/{franquicia}')
def franquicia(franquicia:str):
    '''Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio'''
    try:
        cantidad = int(df[df['collection'] == franquicia]['title'].count())
        ganancia_total = float(df[df['collection'] == franquicia]['revenue'].sum())
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
        respuesta = int(df[df['countries'] == pais]['title'].count())
    except (ValueError, SyntaxError):
        pass 
    return {'pais':pais, 'cantidad':respuesta}


### FUNCION Nro. 5
@app.get('/productoras_exitosas/{productora}')
def productoras_exitosas(productora:str):
    '''Ingresas la productora, entregandote el revenue total y la cantidad de peliculas que realizo '''
    try:
        revenue_total = float(df[df['companies'] == productora]['revenue'].sum())
        cantidad = int(df[df['companies'] == productora]['revenue'].count())
    except (ValueError, SyntaxError):
        pass 
    return {'productora':productora, 'revenue_total': revenue_total,'cantidad':cantidad}



### FUNCION Nro. 6
@app.get('/get_director/{nombre_director}')
def get_director(nombre_director:str):
    try:
        # Buscar todas las filas que contienen el nombre del director en la columna "crew_name"
        PeliculasPorDirector = df[df['crew_name'].str.contains(nombre_director)]

        # Buscar todas las filas que contienen el trabajo de director en la columna "crew_job"
        PeliculasPorCargo = PeliculasPorDirector[PeliculasPorDirector['crew_job'].str.contains('Director')]

        # Seleccionar solo las columnas "title" y "release_year"
        PeliculasConsultadas = PeliculasPorCargo.loc[:, ['title', 'release_year', 'return','budget', 'revenue']]

        # Convertir el DataFrame en una lista de diccionarios
        Lista_De_Dicc = PeliculasConsultadas.to_dict('records')
      
        # Convertir la lista de diccionarios a formato JSON
        Lista_De_Dicc_a_json = json.dumps(Lista_De_Dicc)
        
    except (ValueError, SyntaxError):
        pass 
    return Lista_De_Dicc
  

# ML
@app.get('/recomendacion/{titulo}')
def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    respuesta = "Muy Pronto"
    return {'lista recomendada': respuesta}