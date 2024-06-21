FROM apache/airflow:2.8.1

COPY requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user apache-airflow[virtualenv]==2.8.1 -r /opt/app/requirements.txt
COPY . /opt/app