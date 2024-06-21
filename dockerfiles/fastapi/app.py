import io
import joblib
import mlflow
import pandas as pd
from typing import Literal
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException
from preprocess import preprocess


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

@app.get("/")
async def read_root():
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Spotify API"}))


@app.post("/batch_predict/", response_model=list[ModelOutput])
async def predict(file: UploadFile):
    contents = await file.read()
    print("File read successfully")

    try:
        if not file.content_type.startswith('text/csv'):
            print("Unsupported file format")
            raise HTTPException(415, detail='Unsupported file format. Please upload a CSV file.')

        df = pd.read_csv(io.BytesIO(contents), sep=',')
        print("CSV file parsed successfully")
        print(df.head())

        processed_data = df.apply(preprocess, axis=1)
        print("Data preprocessed successfully")

        predictions = model.predict(processed_data)
        print("Model prediction completed")

        song_predictions = [
            "Liked" if p == 1 else "Did not like"
            for p in predictions.flatten()
        ]

        return [ModelOutput(int_output=bool(p), song_output=song_predictions[i])
                for i, p in enumerate(predictions.flatten())]

    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        raise HTTPException(400, detail='CSV parsing error. Please check the format of your CSV file.')
    except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
        print(f"Model prediction error: {e}")
        raise HTTPException(500, detail='Model prediction error. Please try again later.')
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(500, detail='Internal server error during prediction.')