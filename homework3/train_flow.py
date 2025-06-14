import argparse
from pathlib import Path
import requests
from prefect import flow, task
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn

@task
def download_data(year: int, month: int) -> str:
    month_str = f"{month:02d}"
    filename = f"yellow_tripdata_{year}-{month_str}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"

    if not Path(filename).exists():
        print(f"Downloading {filename} from {url} ...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            raise Exception(f"Failed to download file. HTTP {response.status_code}")
    else:
        print(f"File already exists locally: {filename}")

    return filename

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
    import os
    import mlflow
    import time
    from mlflow.artifacts import download_artifacts

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-prefect")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("intercept", lr.intercept_)

        # Delay to ensure MLflow finishes writing
        time.sleep(2)

        # Dynamically get the full path of model artifact directory
        try:
            local_path = download_artifacts(run_id=run_id, artifact_path="model")
            total_bytes = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, files in os.walk(local_path)
                for f in files
            )

            mlflow.log_metric("model_size_bytes", total_bytes)
            print(f"Model size (bytes): {total_bytes}")

        except Exception as e:
            print(f"Could not compute model size: {e}")

@flow(name="NYC Taxi Model Training Flow")
def main_flow(year: int, month: int):
    file_path = download_data(year, month)
    raw_data = read_data(file_path)
    clean_data = prepare_data(raw_data)
    model, vectorizer = train_model(clean_data)
    log_model(model, vectorizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=int, help="Year of the dataset (e.g. 2023)")
    parser.add_argument("--month", required=True, type=int, help="Month of the dataset (1-12)")
    args = parser.parse_args()

    main_flow(args.year, args.month)