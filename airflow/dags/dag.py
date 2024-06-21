from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests
import datetime

def batch_predict_task():
    url = 'http://localhost:80/batch_predict'
    files = {'file': open('path_to_your_csv_file.csv', 'rb')}
    response = requests.post(url, files=files)
    print(response.json())

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'start_date': days_ago(1),
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

dag = DAG(
    'random_forest_batch_prediction',
    default_args=default_args,
    description='Batch predict using Random Forest model with preprocessing',
    schedule_interval='@daily',
)

batch_predict = PythonOperator(
    task_id='batch_predict_task',
    python_callable=batch_predict_task,
    dag=dag,
)

batch_predict