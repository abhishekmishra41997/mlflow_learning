from mlflow import MlflowClient
from GetData import GetData

if __name__ == '__main__':
    print('Inside main')
    df = GetData().generate_data(base_demand=1000, n_rows=1000)
    print(df.head())
