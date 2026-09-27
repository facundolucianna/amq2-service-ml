import datetime
 
from airflow.decorators import dag, task
 
default_args = {
    'owner': "grupo_taxi",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=45)
}
 
 
@dag(
    dag_id="process_etl_taxi_data",
    description="ETL del dataset de Yellow Taxi NYC para predecir tip_amount",
    default_args=default_args,
    catchup=False,
    tags=["ETL", "taxi"],
)
def process_etl_taxi_data():
 
    @task.virtualenv(
        task_id="get_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_data():
        import awswrangler as wr
        from airflow.models import Variable
 
        df = wr.s3.read_parquet("s3://data/raw/yellow_tripdata_2026-03.parquet")
 
        # nos quedamos solo con pagos con tarjeta, que es donde la propina queda registrada
        df = df[df["payment_type"] == 1].copy()
 
        df["duration_min"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60
 
        extras = ["extra", "mta_tax", "tolls_amount", "improvement_surcharge",
          "congestion_surcharge", "Airport_fee"]
        df["recargos"] = df[[c for c in extras if c in df.columns]].sum(axis=1)
 
        # sacamos viajes con valores no reales
        df = df[
            (df["duration_min"] > 0) & (df["duration_min"] < 180) &
            (df["trip_distance"] > 0) & (df["trip_distance"] < 100) &
            (df["fare_amount"] >= 2.5) &
            (df["tip_amount"] >= 0)
        ]
 
        df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60)
        df = df[df["speed_mph"] <= 100]
 
        sample_size = int(Variable.get("taxi_sample_size", default_var=300000))
        random_state = int(Variable.get("taxi_random_state", default_var=42))
 
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
 
        wr.s3.to_csv(df, "s3://data/taxi/dataset.csv", index=False)
 
    @task.virtualenv(
        task_id="make_dummies_variables",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def make_dummies_variables():
        import awswrangler as wr
        import pandas as pd
        import numpy as np
 
        df = wr.s3.read_csv("s3://data/taxi/dataset.csv", parse_dates=["tpep_pickup_datetime"])
 
        df["hour"] = df["tpep_pickup_datetime"].dt.hour
        df["dow"] = df["tpep_pickup_datetime"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        df["es_hora_pico"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
 
        zonas_aeropuerto = [1, 132, 138]
        df["es_aeropuerto"] = (
            df["PULocationID"].isin(zonas_aeropuerto) |
            df["DOLocationID"].isin(zonas_aeropuerto)
        ).astype(int)
 
        df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"].replace(0, np.nan)
        df["fare_per_mile"] = df["fare_per_mile"].fillna(df["fare_per_mile"].median())
 
        # dummies para las columnas categoricas de baja cardinalidad
        df = pd.get_dummies(df, columns=["VendorID", "RatecodeID"], drop_first=True)
 
        cols_finales = [c for c in df.columns if c not in
                ["tpep_pickup_datetime", "tpep_dropoff_datetime",
                 "payment_type", "hour", "dow", "store_and_fwd_flag"]]
        df = df[cols_finales]
 
        wr.s3.to_csv(df, "s3://data/taxi/dataset_features.csv", index=False)
 
    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.4.2"],
        system_site_packages=True
    )
    def split_dataset():
        import awswrangler as wr
        from airflow.models import Variable
        from sklearn.model_selection import train_test_split
 
        df = wr.s3.read_csv("s3://data/taxi/dataset_features.csv")
 
        test_size = float(Variable.get("taxi_test_size", default_var=0.2))
        random_state = int(Variable.get("taxi_random_state", default_var=42))
 
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
 
        wr.s3.to_csv(train_df, "s3://data/taxi/train.csv", index=False)
        wr.s3.to_csv(test_df, "s3://data/taxi/test.csv", index=False)
 
    @task.virtualenv(
        task_id="normalize_data",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.4.2", "mlflow==3.1.4"],
        system_site_packages=True
    )
    def normalize_data():
        import json
        import boto3
        import awswrangler as wr
        import mlflow
 
        target = "tip_amount"
        numeric_cols = ["duration_min", "trip_distance", "fare_amount", "recargos",
                        "speed_mph", "fare_per_mile"]
        zone_cols = ["PULocationID", "DOLocationID"]
 
        train_df = wr.s3.read_csv("s3://data/taxi/train.csv")
        test_df = wr.s3.read_csv("s3://data/taxi/test.csv")
 
        # target encoding de zonas, calculado solo con train
        global_mean = float(train_df[target].mean())
        pu_means = train_df.groupby("PULocationID")[target].mean().to_dict()
        do_means = train_df.groupby("DOLocationID")[target].mean().to_dict()
 
        for d in (train_df, test_df):
            d["PU_te"] = d["PULocationID"].map(pu_means).fillna(global_mean)
            d["DO_te"] = d["DOLocationID"].map(do_means).fillna(global_mean)
 
        train_df = train_df.drop(columns=zone_cols)
        test_df = test_df.drop(columns=zone_cols)
 
        # estandarizamos usando solo estadisticos de train
        mean_ = train_df[numeric_cols].mean()
        std_ = train_df[numeric_cols].std()
 
        train_df[numeric_cols] = (train_df[numeric_cols] - mean_) / std_
        test_df[numeric_cols] = (test_df[numeric_cols] - mean_) / std_
 
        wr.s3.to_csv(train_df, "s3://data/taxi/train_final.csv", index=False)
        wr.s3.to_csv(test_df, "s3://data/taxi/test_final.csv", index=False)
 
        data_info = {
            "target": target,
            "numeric_cols": numeric_cols,
            "mean": mean_.to_dict(),
            "std": std_.to_dict(),
            "pu_means": {str(k): v for k, v in pu_means.items()},
            "do_means": {str(k): v for k, v in do_means.items()},
            "global_mean": global_mean,
            "columns": [c for c in train_df.columns if c != target],
        }
 
        s3 = boto3.client("s3", endpoint_url="http://s3:9000")
        s3.put_object(
            Bucket="data",
            Key="taxi/data_info/data.json",
            Body=json.dumps(data_info).encode("utf-8"),
        )
 
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("Taxi Tip ETL")
 
        with mlflow.start_run(
            run_name="etl_taxi", experiment_id=experiment.experiment_id
        ):
            mlflow.log_param("rows_train", train_df.shape[0])
            mlflow.log_param("rows_test", test_df.shape[0])
            mlflow.log_param("target", target)
 
    get_data() >> make_dummies_variables() >> split_dataset() >> normalize_data()
 
 
dag = process_etl_taxi_data()
 