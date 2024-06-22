# Aprendizaje de Maquinas II - Despliegue de un modelo

Integrantes:

 - Nicolas Tertusio
 - Leandro Royos
 - Hans Burkli

Implementacion del modelo de random forest desarrollado en Aprendizaje de Maquinas I sobre un dataset de Spotify.

Este dataset contiene datos de las canciones como su energia, tonalidad, ritmo, etc., y predice si a un usuario le gusta o no determinada cancion con determinadas caracteristicas.

Se utilizó FastAPI como front end para subir un archivo .csv (se puede utilizar el archivo data_playlist3.csv en la carpeta test-file para pruebas), y realizar predicciones con el modelo ya entrenado, el cual se exportó como un .pkl.

Al subir un archivo .csv, FastAPI se encarga de subirlo a un bucket de MinIO, el cual luego manda a correr el DAG de prediccion en Apache Airflow.

Todo este sistema se montó en Docker para simular un ambito real.

Para realizar pruebas se debe ingresar a 

> 127.0.0.1:8800

donde se selecciona un archivo .csv y se regresa una tabla con cuantas canciones el modelo predice que le gustara al usuario y cuantas no.

Para recibir los datos individualmente (en vez de agrupados, como se explicó anteriormente), se deben descomentar las siguientes lineas en el archivo app.py (dentro de /dockerfiles/fastapi):

> #return [ModelOutput(int_output=bool(p), song_output=song_predictions[i]) for i, p in enumerate(predictions.flatten())]

![](/test-file/fastapi.PNG)

