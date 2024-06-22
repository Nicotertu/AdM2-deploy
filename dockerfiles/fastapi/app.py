import io
import joblib
import mlflow
import pandas as pd
import os
import requests
from typing import Literal
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException
from preprocess import preprocess
from minio import Minio

def load_model(model_name: str, alias: str):
    try:
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
    except:
        model_ml = joblib.load('model/random_forest_model.pkl')

    return model_ml

class ModelInput(BaseModel):
    acousticness: float = Field(
        description="Una medida de qué tan acústica es una pista. Cuanto más cercano a 1, más acústica es la pista.")
    danceability: float = Field(
        description="Una medida de qué tan adecuada es una pista para bailar, basada en una combinación de elementos musicales como el tempo, la estabilidad del ritmo, la fuerza del ritmo y la regularidad general.")
    duration: int = Field(
        description="La duración de la pista en milisegundos.")
    energy: float = Field(
        description="Representa la intensidad y actividad percibida de una pista. Las pistas energéticas suelen tener ritmos más rápidos y mayor intensidad percibida.")
    instrumentalness: float = Field(
        description="Una medida de qué tan instrumental es una pista. Cuanto más cercano a 1, mayor es la probabilidad de que la pista no contenga voces.")
    key: int = Field(
        description="La tonalidad de la pista expresada como un número entero. Puede tomar valores del 0 al 11, representando diferentes tonalidades musicales.")
    liveness: float = Field(
        description="Una medida de qué tan probable es que la pista haya sido grabada en vivo. Cuanto más cercano a 1, más probable es que la pista sea en vivo.")
    loudness: float = Field(
        description="La intensidad de la pista en decibeles (dB).")
    mode: int = Field(
        description="Indica si la pista está en tonalidad mayor (0) o menor (1).")
    speechiness: float = Field(
        description="Una medida de qué tan hablada es una pista en comparación con ser puramente instrumental. Valores superiores a 0.66 suelen indicar que la pista es puramente hablada.")
    tempo: float = Field(
        description="El ritmo de la pista en pulsos por minuto (BPM).")
    time_signature: int = Field(
        description="El compás de la pista, es decir, el número de pulsos en un compás musical.")
    valence: float = Field(
        description="Una medida de la positividad o negatividad de una pista. Cuanto más cercano a 1, más positiva es la pista.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "acousticness": 0.713,
                    "danceability": 0.514,
                    "duration": 100125,
                    "energy": 0.521,
                    "instrumentalness": 0.816,
                    "key": 8,
                    "liveness": 0.112,
                    "loudness": -14.835,
                    "mode": 0,
                    "speechiness": 0.0444,
                    "tempo": 119.879,
                    "time_signature": 4,
                    "valence": 0.143
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    int_output: int = Field(
        description="Output of the model. 0 means user did not like the song, 1 means user liked the song",
    )
    song_output: Literal["Liked", "Did not like"] = Field(
        description="Output of the model in string form",
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 1,
                    "str_output": "Liked",
                }
            ]
        }
    }


model = load_model("random_forest_model", "lovely")

app = FastAPI()

minio_client = Minio(
    "minio:9000",
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

def trigger_dag(file_location: str):
    airflow_url = "http://airflow-webserver:8080/api/v1/dags/Spotify_batch_prediction_minio/dagRuns"
    data = {
        "conf": {"file_location": file_location}
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer Ii9vcHQvYWlyZmxvdy9kYWdzL3Nwb3RpZnlfYmF0Y2hfcHJlZGljdF9taW5pby5weSI.X4OL1O2o4V_z32WBYn06bkRv_vs"
    }
    response = requests.post(airflow_url, json=data, headers=headers)
    print(data)
    print(headers)
    print(response)

    response.raise_for_status()

@app.get("/")
async def read_root():
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>File Upload</title>
        </head>
        <body>
            <h1>Upload a CSV File</h1>
            <form id="upload-form" action="/batch_predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file_upload" accept=".csv" required>
                <button type="submit">Upload</button>
            </form>
            <div id="result"></div>

            <script>
                document.getElementById('upload-form').addEventListener('submit', async function(event) {
                    event.preventDefault();
                    const formData = new FormData(this);
                    const response = await fetch(this.action, {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.text();
                    document.getElementById('result').innerHTML = result;
                });
            </script>
        </body>
        </html>
        """)


@app.post("/batch_predict/", response_model=list[ModelOutput])
async def predict2(file_upload: UploadFile):
    contents = await file_upload.read()
    print("File read successfully")

    if not file_upload.content_type.startswith('text/csv'):
        print("Unsupported file format")
        raise HTTPException(415, detail='Unsupported file format. Please upload a CSV file.')
    
    print(f"minio {minio_client}")
    bucket_name = "data"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    minio_client.put_object(bucket_name, file_upload.filename, io.BytesIO(contents), len(contents))
    print("File uploaded to minio successfully")

    #trigger_dag(file_location)
    #return {"file_location": file_location}

    try:
        df = pd.read_csv(io.BytesIO(contents), sep=',')
        print("CSV file parsed successfully")
        print(df.head())

        #processed_data = df.apply(preprocess, axis=1)
        processed_data = preprocess(df)
        print("Data preprocessed successfully")

        predictions = model.predict(processed_data)
        print("Model prediction completed")

        song_predictions = [
            "Liked" if p == 1 else "Did not like"
            for p in predictions.flatten()
        ]

        liked_count = sum(predictions)
        disliked_count = len(predictions) - liked_count

        # For individual results
        #return [ModelOutput(int_output=bool(p), song_output=song_predictions[i])
        #        for i, p in enumerate(predictions.flatten())]

        # For grouped results
        html_start = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Results</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Prediction Results</h1>
            <table border="1">
                <tr>
                    <th>Liked songs</th>
                    <th>Disliked songs</th>
                </tr>
                <tr>
                    <td>"""
        html_part1 = f"{liked_count}".strip()
        html_part2 = """</td>
                    <td>"""
        html_part3 = f"{disliked_count}".strip()
        html_part4 = """</td>
                </tr>
            </table>
            <canvas id="pieChart" width="400" height="400"></canvas>
            <script>
                var ctx = document.getElementById('pieChart').getContext('2d');
                var chart = new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['Liked', 'Did not like'],
                        datasets: [{
                            data: [""".strip()

        html_data = f"{liked_count}, {disliked_count}".strip()

        html_end = """
                            ],
                            backgroundColor: ['#36a2eb', '#ff6384']
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            </script>
        </body>
        </html>
        """.strip()

        html_content = html_start + html_part1 + html_part2 + html_part3 + html_part4 + html_data + html_end
        return HTMLResponse(content=html_content)

    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        raise HTTPException(400, detail='CSV parsing error. Please check the format of your CSV file.')
    except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
        print(f"Model prediction error: {e}")
        raise HTTPException(500, detail='Model prediction error. Please try again later.')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(500, detail='Internal server error during prediction.')


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/batch_predict/", response_class=HTMLResponse)
async def serve_index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <h1>Upload a CSV File</h1>
        <form id="upload-form" action="/batch_predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="file_upload" accept=".csv" required>
            <button type="submit">Upload</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('upload-form').addEventListener('submit', async function(event) {
                event.preventDefault();
                const formData = new FormData(this);
                const response = await fetch(this.action, {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                document.getElementById('result').textContent = JSON.stringify(result, null, 2);
            });
        </script>
    </body>
    </html>
    """