class CreateNewExperiment:
    def __init__(self,client):
        self.client=client


    def create_experiment(self,exp_name='Test',experiment_tags = {"project_name": "MLFlow testing","store_dept": "produce","team": "Abhishek","mlflow.note.content": "This is test experiment"}):


        """
        :param exp_name: Name of the experiment
        :param experiment_tags: Tags to define
        :return: experiment object
        """

        self.client.create_experiment(name=exp_name, tags=experiment_tags)


