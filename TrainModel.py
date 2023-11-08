import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class TrainModel:
    def __init__(self,tracking_uri="http://127.0.0.1:8080"):
        self.tracking_uri=tracking_uri
    def tain_model(self,data,experiment_name,run_name,artifact_path):
        mlflow.set_tracking_uri(self.tracking_uri)

        apple_experiment = mlflow.set_experiment(experiment_name)

        X = data.drop(columns=["date", "demand"])
        y = data["demand"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "bootstrap": True,
            "oob_score": False,
            "random_state": 888,
        }

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)

        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

        with mlflow.start_run(run_name=run_name) as run:
            # Log the parameters used for the model fit
            mlflow.log_params(params)

            # Log the error metrics that were calculated during validation
            mlflow.log_metrics(metrics)

            # Log an instance of the trained model for later use
            mlflow.sklearn.log_model(sk_model=rf, input_example=X_val, artifact_path=artifact_path)

