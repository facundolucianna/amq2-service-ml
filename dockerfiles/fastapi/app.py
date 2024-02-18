import json
import pickle
import boto3
import mlflow
import hashlib

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from copy import deepcopy


# ATTENTION: This is a bad practice, just intended for didactic reason, never use
# password in plain text. This password could be in a DB o a secret vault, like
# GitHub Secrets or even better implement an API token system.
hashed_pass = hashlib.md5("qwerty".encode())

try:
    # Load the trained model
    model_name = "heart_disease_model_prod"
    alias = "champion"

    mlflow.set_tracking_uri('http://0.0.0.0:5001')
    client = mlflow.MlflowClient()

    model_data = client.get_model_version_by_alias(model_name, alias)
    model = mlflow.sklearn.load_model(model_data.source)
except:
    # If there is no registry in MLflow, open the dafault model
    file = open('/app/files/model.pkl', 'rb')
    model = pickle.load(file)
    file.close()

try:
    # Load information of the ETL pipeline
    client = boto3.client('s3')

    client.head_object(Bucket='data', Key='data_info/data.json')
    result = client.get_object(Bucket='data', Key='data_info/data.json')
    text = result["Body"].read().decode()
    data_dict = json.loads(text)

    data_dict["standard_scaler_mean"] = np.array(data_dict["standard_scaler_mean"])
    data_dict["standard_scaler_std"] = np.array(data_dict["standard_scaler_std"])
except:
    file = open('/app/files/data.json', 'r')
    data_dict = json.load(file)
    file.close()


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Detector API"}


@app.post("/predict/")
def predict(data: dict):

    if 'features' not in data:
        raise HTTPException(status_code=400, detail="Features data is missing")

    features = data.get('features')
    if not isinstance(features, list):
        raise HTTPException(status_code=400, detail="Features should be a list")

    expected_number_of_features = len(data_dict["columns"])

    if len(features) != expected_number_of_features:
        raise HTTPException(status_code=400, detail=f"Expected {expected_number_of_features} features")

    try:
        features = np.array(features).reshape(1, -1)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid feature data")

    # Process the features
    features_serie = pd.Series(features, index=data_dict["columns"])

    features_serie = pd.get_dummies(data=features_serie,
                                    columns=data_dict["categorical_columns"],
                                    drop_first=True)

    # Scale the data
    features_serie = (features_serie-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Make the prediction
    prediction = model.predict(features_serie)

    str_pred = "Healthy patient"
    if prediction[0] > 0:
        str_pred = "Heart disease detected"

    return {"int_output": prediction[0], "str_output": str_pred}


@app.get("/schema/")
def predict(data: dict):

    input_schema = data_dict["columns_dtypes"]
    output_schema = {
        'int_output': 'int64',
        'str_output': 'string'
    }

    return {"input_schema": input_schema, "output_schema": output_schema}


@app.put("/refresh_model/")
def refresh_model(data):
    global model
    global data_dict

    if 'pass' not in data:
        raise HTTPException(status_code=403, detail="Not authorized")

    password = data.get('pass')
    if not isinstance(password, str):
        raise HTTPException(status_code=400, detail="Pass should be a string")

    hashed_user_pass = hashlib.md5(password.encode())

    if hashed_user_pass != hashed_pass:
        raise HTTPException(status_code=403, detail="Invalid password")

    try:
        # Load the trained model
        mlflow.set_tracking_uri('http://0.0.0.0:5001')
        client_new = mlflow.MlflowClient()

        model_data_new = client_new.get_model_version_by_alias(model_name, alias)
        model_new = mlflow.sklearn.load_model(model_data_new.source)
    except:
        raise HTTPException(status_code=404, detail="No new model in the registry")

    try:
        # Load information of the ETL pipeline
        client_new = boto3.client('s3')

        client_new.head_object(Bucket='data', Key='data_info/data.json')
        result_new = client_new.get_object(Bucket='data', Key='data_info/data.json')
        text_new = result_new["Body"].read().decode()
        data_dict_new = json.loads(text_new)

        data_dict_new["standard_scaler_mean"] = np.array(data_dict_new["standard_scaler_mean"])
        data_dict_new["standard_scaler_std"] = np.array(data_dict_new["standard_scaler_std"])
    except:
        raise HTTPException(status_code=404, detail="No new model in the registry")

    data_dict = data_dict_new.copy()
    model = deepcopy(model_new)
    return {"message": "Ok"}
