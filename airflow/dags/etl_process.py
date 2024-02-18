import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Heart Disease Data

This DAG extracts information from the original CSV file stored in the UCI Machine Learning Repository of the 
[Heart Disease repository](https://archive.ics.uci.edu/dataset/45/heart+disease). 
It preprocesses the data by creating dummy variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Facundo Adrian Lucianna",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_heart_data",
    description="ETL process for heart data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Heart Disease"],
    default_args=default_args,
    catchup=False,
)
def process_etl_heart_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["ucimlrepo==0.0.3",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw data from UCI repository
        """
        import awswrangler as wr
        from ucimlrepo import fetch_ucirepo
        from airflow.models import Variable

        # fetch dataset
        heart_disease = fetch_ucirepo(id=45)

        data_path = "s3://data/raw/heart.csv"
        dataframe = heart_disease.data.original

        target_col = Variable.get("target_col_heart")

        # Replace level of heart decease to just distinguish presence 
        # (values 1,2,3,4) from absence (value 0).
        dataframe.loc[dataframe[target_col] > 0, target_col] = 1

        wr.s3.to_csv(df=dataframe,
                     path=data_path,
                     index=False)


    @task.virtualenv(
        task_id="make_dummies_variables",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def make_dummies_variables():
        """
        Convert categorical variables into one-hot encoding.
        """
        import datetime
        import awswrangler as wr
        import pandas as pd
        import mlflow

        from airflow.models import Variable

        data_original_path = "s3://data/raw/heart.csv"
        data_end_path = "s3://data/raw/heart_dummies.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)
        # Drop NaN
        dataset.dropna(inplace=True, ignore_index=True)

        # Force type in categorical columns
        dataset["cp"] = dataset["cp"].astype(int)
        dataset["restecg"] = dataset["restecg"].astype(int)
        dataset["slope"] = dataset["slope"].astype(int)
        dataset["ca"] = dataset["ca"].astype(int)
        dataset["thal"] = dataset["thal"].astype(int)

        categories_list = ["cp", "restecg", "slope", "ca", "thal"]
        dataset_with_dummies = pd.get_dummies(data=dataset,
                                              columns=categories_list,
                                              drop_first=True)

        wr.s3.to_csv(df=dataset_with_dummies,
                     path=data_end_path,
                     index=False)

        mlflow.set_tracking_uri('http://192.168.0.21:5001')
        experiment = mlflow.set_experiment("Heart Disease")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Heart disease"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(dataset_with_dummies,
                                                 source="https://archive.ics.uci.edu/dataset/45/heart+disease",
                                                 targets=Variable.get("target_col_heart"),
                                                 name="heart_data_complete")
        mlflow.log_input(mlflow_dataset, context="Dataset")


    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
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

        data_original_path = "s3://data/raw/heart_dummies.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = Variable.get("test_size_heart")
        target_col = Variable.get("target_col_heart")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        save_to_csv(X_train, "s3://data/final/train/heart_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/heart_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/heart_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/heart_y_test.csv")

    @task.virtualenv(
        task_id="normalize_numerical_features",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def normalize_data():
        """
        Standardization of numerical columns
        """
        import awswrangler as wr
        import pandas as pd
        import mlflow

        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/heart_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/heart_X_test.csv")

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        save_to_csv(X_train, "s3://data/final/train/heart_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/heart_X_test.csv")

        mlflow.set_tracking_uri('http://192.168.0.21:5001')
        experiment = mlflow.set_experiment("Heart Disease")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)


    get_data() >> make_dummies_variables() >> split_dataset() >> normalize_data()


dag = process_etl_heart_data()