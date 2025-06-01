from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

@task
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Initial rows loaded: {len(df)}")
    return df

@task
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Filtered rows: {len(df)}")
    return df

@task
def train_model(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"Intercept: {lr.intercept_:.2f}")

    return lr, dv

@task
def log_model(lr, dv):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-prefect")

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("intercept", lr.intercept_)

        print("Model logged to MLflow")

@flow(name="NYC Taxi Model Training Flow")
def main_flow(file_path: str):
    raw_data = read_data(file_path)
    clean_data = prepare_data(raw_data)
    model, vectorizer = train_model(clean_data)
    log_model(model, vectorizer)

if __name__ == "__main__":
    main_flow("yellow_tripdata_2023-03.parquet")