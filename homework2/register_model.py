import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt-v2"
EXPERIMENT_NAME = "random-forest-best-models-v2"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
#mlflow.sklearn.autolog()

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        try:
            # Safely parse and cast hyperparameters
            new_params = {param: int(params[param]) for param in RF_PARAMS if param in params}
            rf = RandomForestRegressor(**new_params)
            rf.fit(X_train, y_train)

            # Evaluate and log metrics
            val_preds = rf.predict(X_val)
            test_preds = rf.predict(X_test)

            val_rmse = root_mean_squared_error(y_val, val_preds)
            test_rmse = root_mean_squared_error(y_test, test_preds)

            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.sklearn.log_model(rf, "model")

            return run.info.run_id, test_rmse

        except Exception as e:
            print(f"[ERROR] Failed to train model with params {params}: {e}")
            mlflow.end_run(status="FAILED")
            return None, float("inf")

@click.command()
@click.option("--data_path", default="./output", help="Location of processed data")
@click.option("--top_n", default=5, type=int, help="Number of top models to evaluate")
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        print(f"[ERROR] Experiment '{HPO_EXPERIMENT_NAME}' not found.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    best_run_id = None
    best_rmse = float("inf")

    for run in runs:
        print(f"→ Evaluating HPO run {run.info.run_id}...")
        run_id, test_rmse = train_and_log_model(data_path=data_path, params=run.data.params)
        print(f"   Test RMSE: {test_rmse:.4f}" if run_id else "   Skipped due to error")

        if run_id and test_rmse < best_rmse:
            best_rmse = test_rmse
            best_run_id = run_id

    if best_run_id:
        print(f"\nBest run: {best_run_id} with test RMSE: {best_rmse:.4f}")
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "RandomForestRegressor_BestModel"
        mlflow.register_model(model_uri=model_uri, name=model_name)
    else:
        print("\nNo successful runs found. Model not registered.")

if __name__ == '__main__':
    run_register_model()
