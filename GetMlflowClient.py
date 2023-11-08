from mlflow import MlflowClient

class GetMlflowClient:
    def __init__(self):
        self.tracking_uri="http://127.0.0.1:8080"

    def initialize_client(self):
        client = MlflowClient(self.tracking_uri)
        return client

    def get_experiment_details(self,client,filter_string=None):

        all_experiments = client.search_experiments(filter_string)
        return all_experiments


