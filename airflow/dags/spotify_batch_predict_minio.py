import requests
import datetime
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from minio import Minio

def batch_predict_task(**kwargs):
    minio_client = Minio(
        "minio:9000",
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False
    )

    file_location = kwargs['dag_run'].conf.get('file_location', '')
    if not file_location:
        print("No file location provided!")
        return
    
    bucket_name, object_name = file_location.split('/', 1)

    response = minio_client.get_object(bucket_name, object_name)
    file_path = f"/tmp/{object_name}"
    with open(file_path, "wb") as file_data:
        for d in response.stream(32 * 1024):
            file_data.write(d)

    url = 'http://localhost:8800/batch_predict'
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, files=files)
    print(response.json())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

dag = DAG(
    'Spotify_batch_prediction_minio',
    default_args=default_args,
    description='Batch predict using Random Forest model with preprocessing',
    schedule_interval=None,
    start_date=days_ago(1),
)

batch_predict = PythonOperator(
    task_id='batch_predict_task',
    python_callable=batch_predict_task,
    provide_context=True,
    dag=dag,
)

batch_predict