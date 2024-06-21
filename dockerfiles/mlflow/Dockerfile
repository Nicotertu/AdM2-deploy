FROM python:3.11-slim

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --no-cache-dir -r /opt/app/requirements.txt
RUN apt-get update && apt-get install -y wget
RUN pip install --upgrade pip
RUN pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql

COPY . /opt/app

EXPOSE 5000