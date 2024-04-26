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


    @task.virtualenv(
        task_id="impute_missing_values",
        requirements=["pandas",
                      "scikit-learn",
                      "numpy",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )   
    def impute_missing_values():
        """
        Multiple Imputation by Chained Equations (MICE).
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow
        
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        
        from airflow.models import Variable
        
        data_original_path = "s3://data/raw/water-quality.csv"
        data_end_path = "s3://data/raw/water-quality-imputed.csv"
        
        dataset = wr.s3.read_csv(data_original_path)
        
        NUMERICAL_FEATURES = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes',  'Turbidity']
        TARGET = Variable.get("target_col_water")
        
        X = dataset[NUMERICAL_FEATURES].values
        y = dataset[TARGET].values

        # Crear el IterativeImputer con el parámetro imputer__n_nearest_features igual a 6
        mice_imputer = IterativeImputer(n_nearest_features=6, random_state=0)
        mice_imputer.fit(X, y)
        X_imputed = mice_imputer.transform(X)
        y = y.reshape(-1, 1)
        
        dataframe_imputed = pd.DataFrame(data=np.concatenate((X_imputed, y), axis=1), columns=dataset.columns)
        
        wr.s3.to_csv(df=dataframe_imputed,
                     path=data_end_path,
                     index=False)
        
        # Save information of the dataset
        client = boto3.client('s3')
        
        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e
            
        dataset_log = dataset.drop(columns=TARGET)
        dataset_imputed_log = dataframe_imputed.drop(columns=TARGET)
        
        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_imputation'] = dataset_imputed_log.columns.to_list()
        data_dict['target_col'] = TARGET
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_imputation'] = {k: str(v) for k, v in dataset_imputed_log.dtypes.to_dict().items()}

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )
        
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Water Quality")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Water Quality"},
                         log_system_metrics=True)
        
        mlflow_dataset = mlflow.data.from_pandas(dataset,
                                                 source="https://www.kaggle.com/datasets/adityakadiwal/water-potability",
                                                 targets=TARGET,
                                                 name="water_quality_complete")
        mlflow_dataset_dummies = mlflow.data.from_pandas(dataframe_imputed,
                                                         source="https://www.kaggle.com/datasets/adityakadiwal/water-potability",
                                                         targets=TARGET,
                                                         name="water_quality_complete_imputed")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")


    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        data_original_path = "s3://data/raw/water-quality-imputed.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = Variable.get("test_size_water")
        target_col = Variable.get("target_col_water")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, stratify=y)

        save_to_csv(X_train, "s3://data/final/train/water_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/water_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/water_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/water_y_test.csv")

    @task.virtualenv(
        task_id="normalize_numerical_features",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def normalize_data():
        """
        Standardization of numerical columns
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/water_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/water_X_test.csv")

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = scaler.fit_transform(X_train)
        X_test_arr = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        save_to_csv(X_train, "s3://data/final/train/water_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/water_X_test.csv")

        # Save information of the dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = scaler.mean_.tolist()
        data_dict['standard_scaler_std'] = scaler.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Water Quality")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", scaler.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", scaler.mean_)
            mlflow.log_param("Standard Scaler scale values", scaler.scale_)


    get_data() >> impute_missing_values() >> split_dataset() >> normalize_data()
    
dag = process_etl_water_data()