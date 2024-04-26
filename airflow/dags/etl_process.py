import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Water Quality

This DAG extracts information from the original CSV file stored in the Kaggle Dataset's Repository of the 
[Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability). 
It preprocesses the data by imputing missing values and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 75/25 and they are stratified.
"""

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="process_etl_water_data",
    description="ETL process for water quality data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Water Quality"],
    default_args=default_args,
    catchup=False,
)
def process_etl_water_data():
    
    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["kaggle==1.6.3",
                      "pandas",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )    
    def get_data():
        """
        Load the raw data from Kaggle repository
        """
        import awswrangler as wr
        from kaggle.api.kaggle_api_extended import KaggleApi
        import pandas as pd
        
        # Configuracion de la API de Kaggle
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = "adityakadiwal/water-potability"
        file_name="water_potability.csv"
        data_path = "./data"
        
        # Descarga del dataset desde Kaggle
        api.dataset_download_file(dataset_name, file_name, path=data_path)
        
        dataframe = pd.read_csv(f"{data_path}/{file_name}")
        wr.s3.to_csv(df=dataframe,
                     path="s3://data/raw/water-quality.csv",
                     index=False)


    get_data()
    
dag = process_etl_water_data()