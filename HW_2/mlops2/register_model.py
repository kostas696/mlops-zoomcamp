import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        new_params = {param: int(params[param]) for param in RF_PARAMS}
        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log the model
        mlflow.sklearn.log_model(rf, "model")

        return run.info.run_id, test_rmse

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs from the hyperparameter optimization experiment
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    best_run_id = None
    best_rmse = float("inf")

    # Evaluate and log the models on the test set
    for run in runs:
        run_id, test_rmse = train_and_log_model(data_path=data_path, params=run.data.params)
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_run_id = run_id

    print(f"Best run ID: {best_run_id}, Best test RMSE: {best_rmse}")

    # Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "RandomForestRegressor_BestModel"
    mlflow.register_model(model_uri=model_uri, name=model_name)

if __name__ == '__main__':
    run_register_model()