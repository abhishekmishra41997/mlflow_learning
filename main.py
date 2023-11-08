from mlflow import MlflowClient
from GetData import GetData
from GetMlflowClient import GetMlflowClient
from CreateNewExperiment import CreateNewExperiment
from TrainModel import TrainModel

if __name__ == '__main__':
    print('Inside main')

    tracking_uri = "http://127.0.0.1:8080"
    filter_string = 'experiment.name == "Default"'

    experiment_description = (
        "This is my first mlflow run for mlflow learning"
    )

    experiment_tags = {
        "project_name": "Mlflow-first-run",
        "store_dept": "produce",
        "team": "Abhishek",
        "project_quarter": "Q3",
        "mlflow.note.content": experiment_description,
    }

    run_name = "apples_rf_test"

    artifact_path = "rf_apples"

    experiment_name='First-Experiment-1'

    df = GetData().generate_data(base_demand=1000, n_rows=1000)

    client=GetMlflowClient().initialize_client()
    exp_detail=GetMlflowClient().get_experiment_details(client)
    print('Experimnt detail-->',exp_detail)
    CreateNewExperiment(client).create_experiment(experiment_name,experiment_tags=experiment_tags)

    TrainModel().tain_model(df,experiment_name,run_name,artifact_path)


    #print(df.head())
