import os
from metaflow import FlowSpec, step, S3

os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"


class BatchProcessingModel(FlowSpec):

    @step
    def start(self):
        print("Starting Batch Prediction")
        self.next(self.load_data, self.load_model)

    @step
    def load_data(self):
        import pandas as pd

        s3 = S3(s3root="s3://batch/")
        data_obj = s3.get("data/iris.csv")
        self.X_batch = pd.read_csv(data_obj.path)
        self.next(self.batch_processing)

    @step
    def load_model(self):
        from xgboost import XGBClassifier

        s3 = S3(s3root="s3://batch/")
        model_param = s3.get("artifact/model.json")

        loaded_model = XGBClassifier()
        loaded_model.load_model(model_param.path)

        self.model = loaded_model
        self.next(self.batch_processing)

    def train_xgb(self):

        print(self.model.predict(self.X))
        self.next(self.end)

    @step
    def batch_processing(self, previous_tasks):
        import hashlib

        print("Obtaining predictions")

        for task in previous_tasks:
            if hasattr(task, 'X_batch'):
                data = task.X_batch
            if hasattr(task, 'model'):
                model = task.model

        out = model.predict(data)

        data['key'] = data.apply(lambda row: ' '.join(map(str, row)), axis=1)
        data['hashed'] = data['key'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

        dict_redis = {}
        for index, row in data.iterrows():
            dict_redis[row["hashed"]] = int(out[index])

        self.redis_data = dict_redis

        self.next(self.ingest_redis)

    @step
    def ingest_redis(self):
        import redis

        print("Ingesting Redis")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        for key in self.redis_data:
            r.set(key, self.redis_data[key])
        self.next(self.end)

    @step
    def end(self):
        print("Finished")


if __name__ == "__main__":
    BatchProcessingModel()
