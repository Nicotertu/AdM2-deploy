FROM python:3.11-slim

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /opt/app/requirements.txt
COPY . /opt/app

EXPOSE 8800