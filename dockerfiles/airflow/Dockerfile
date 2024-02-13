FROM apache/airflow:latest
COPY requirements.txt /requirements.txt

RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user apache-airflow==${AIRFLOW_VERSION} -r /requirements.txt
