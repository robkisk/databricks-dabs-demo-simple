# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC
# MAGIC # Models in Unity Catalog - NYC Taxi Trip Duration Model
# MAGIC
# MAGIC This is the base notebook of our project and is used to demonstrate a simple model training pipeline, where we predict the duration of [taxi trips in New York City](https://www.kaggle.com/c/nyc-taxi-trip-duration). We started our project with this notebook and proceeded to refactor and modularize it into a python package to be deployed in an end to end MLOps workflow.
# MAGIC
# MAGIC The core aim of this notebook is to demonstrate how to register a model to Unity Catalog, and subsequently load the model for inference. The wider repo in which this notebook sits aims to demonstrate how to go from this notebook to a productionized ML application, using Unity Catalog to manage our registered model.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC The following requirements are needed in order to be able to register ML models in Unity Catalog:
# MAGIC
# MAGIC * A cluster running Databricks Runtime 13.0 or above with access to Unity Catalog ([AWS](https://docs.databricks.com/data-governance/unity-catalog/compute.html#create-clusters--sql-warehouses-with-unity-catalog-access)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/compute))
# MAGIC * Permissions to create models in at least one Unity Catalog schema. In particular, you need `USE CATALOG` permissions on the parent catalog, and both `USE SCHEMA` and `CREATE MODEL` permissions on the parent schema. If you hit permissions errors while running the example below, ask a catalog/schema owner or admin for access.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC The notebook contains the following steps:
# MAGIC
# MAGIC 1. **Imports**: Necessary libraries and modules are imported. MLflow autologging is enabled and the registry URI is set to "databricks-uc".
# MAGIC 1. **Global Variables**: Set global variables to be used throughout the notebook.
# MAGIC 1. **Load Data**: Load the NYC Taxi Trip Duration dataset.
# MAGIC 1. **Split Data**: The loaded data is split into training, validation, and test sets.
# MAGIC 1. **Feature Engineering**: The input DataFrame is extended with additional features, and unneeded columns are dropped. Define a Scikit-Learn pipeline to perform feature engineering.
# MAGIC 1. **Train Model**: Train an XGBoost Regressor model, tracking parameters, metrics and model artifacts to MLflow.
# MAGIC 1. **Register Model**: The trained model is registered to Unity Catalog. Update the registered model with a "Champion" alias.
# MAGIC 1. **Consume Model**: The "Champion" version of the registered model is loaded and used for inference against the test dataset.

# COMMAND ----------
import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Needed for working remotely with Databricks MLFlow and Unity Catalog
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.autolog()

# COMMAND ----------
# Global Variables
EXPERIMENT_PATH = "<path_to_experiment>"
CATALOG_NAME = "<catalog_name>"
SCHEMA_NAME = "<schema_name>"
MODEL_NAME = "<model_name>"
REGISTERED_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"

# COMMAND ----------
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------
# Example raw input data from Delta
# Change the path to the location of your data
from get_spark import GetSpark

# This leverages DatabricksSession and dbconnect to connect to a Databricks cluster
spark = GetSpark().init_spark(eager=True)

nyc_taxi_pdf = (
    spark.read.format("delta")
    .load("dbfs:/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
    .toPandas()
)
nyc_taxi_pdf


# COMMAND ----------
def split_data(
    pdf: pd.DataFrame, target_column: str, split_ratio: tuple = (0.7, 0.2, 0.1)
) -> tuple:
    """Split the data into a training set, validation set, and a test set.

    Args:
        pdf (pd.DataFrame): Input data.
        target_column (str): Name of the target column.
        split_ratio (tuple): A tuple that specifies the ratio of the training, validation, and test sets.

    Returns:
        tuple: A tuple containing the features and target for the training, validation, and test sets.
    """
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1"

    X = pdf.drop(target_column, axis=1)
    y = pdf[target_column]

    # Calculate split sizes
    train_size, val_size = split_ratio[0], split_ratio[0] + split_ratio[1]

    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=split_ratio[2], random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size / (train_size + val_size),
        random_state=42,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Feature engineering
# MAGIC
# MAGIC We create the following functions for feature engineering: `calculate_features()` and `transformer_fn()`.
# MAGIC
# MAGIC - `calculate_features()`: used to extend the input DataFrame with pickup day of the week and hour, and trip duration.
# MAGIC - `transformer_fn()`: returns an unfitted transformer that defines `fit()` and `transform()` methods which perform feature engineering and encoding.
# MAGIC
# MAGIC We will use the resulting `transformer_fn()` as part of our sklearn `Pipeline`.


# COMMAND ----------
def calculate_features(pdf: pd.DataFrame) -> pd.DataFrame:
    """Function to conduct feature engineering.

    Extend the input dataframe with pickup day of week and hour, and trip duration.
    Drop the now-unneeded pickup datetime and dropoff datetime columns.

    Args:
        pdf (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    pdf["pickup_dow"] = pdf["tpep_pickup_datetime"].dt.dayofweek
    pdf["pickup_hour"] = pdf["tpep_pickup_datetime"].dt.hour
    trip_duration = pdf["tpep_dropoff_datetime"] - pdf["tpep_pickup_datetime"]
    pdf["trip_duration"] = trip_duration.map(lambda x: x.total_seconds() / 60)
    pdf.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True)
    return pdf


def transformer_fn() -> Pipeline:
    """Define sklearn pipeline.

    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.

    Returns:
        sklearn.pipeline.Pipeline: Unfitted sklearn transformer
    """
    return Pipeline(
        steps=[
            (
                "calculate_time_and_duration_features",
                FunctionTransformer(calculate_features, feature_names_out=None),
            ),
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "hour_encoder",
                            OneHotEncoder(categories="auto"),
                            ["pickup_hour"],
                        ),
                        (
                            "day_encoder",
                            OneHotEncoder(categories="auto"),
                            ["pickup_dow"],
                        ),
                        (
                            "std_scaler",
                            StandardScaler(),
                            ["trip_distance", "trip_duration"],
                        ),
                    ]
                ),
            ),
        ]
    )


# COMMAND ----------
# MAGIC %md
# MAGIC
# MAGIC ## Train model
# MAGIC
# MAGIC ML model versions in UC must have a model signature. If you’re not already logging MLflow models with signatures in your model training workloads, you can either:
# MAGIC
# MAGIC 1. Use MLflow autologging
# MAGIC     - MLflow autologing automatically logs models when they are trained in a notebook. Model signature is inferred and logged alongside the model artifacts.
# MAGIC    - Read https://mlflow.org/docs/latest/tracking.html#automatic-logging to see if your model flavor is supported.
# MAGIC 2. Manually set the model signature in `mlflow.<flavor>.log_model`
# MAGIC     - Infer model signature via [`mlflow.models.infer_signature`](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.infer_signature), or manually define the signature.
# MAGIC     - Pass the model signature to `log_model` via the `signature` argument
# MAGIC
# MAGIC Given that we have enabled MLflow autologging at the outset of the notebook we will not need to explcitly set the model signature.
# MAGIC
# MAGIC In the following cell:
# MAGIC
# MAGIC - `estimator_fn()` defines an unfitted `XGBRegressor` estimator that defines (using the sklearn API). This is subsequently used as the estimator in our sklearn `Pipeline`.
# MAGIC - `train_model` creates and fits our sklearn `Pipeline`, tracking to MLflow.


# COMMAND ----------
def estimator_fn(*args, **kwargs) -> BaseEstimator:
    """Define XGBRegressor model.

    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.

    Returns:
        sklearn.base.BaseEstimator: Unfitted sklearn base estimator
    """
    return XGBRegressor(objective="reg:squarederror", random_state=42, *args, **kwargs)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """Function to trigger model training, tracking to MLflow Tracking.

    Create a pipeline that includes feature engineering and model training, and fit it on the training data.
    Return the run_id of the MLflow run.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data target.

    Returns:
        str: MLflow run_id.
    """
    with mlflow.start_run():
        pipeline = Pipeline(
            steps=[("transformer", transformer_fn()), ("model", estimator_fn())]
        )

        pipeline.fit(X_train, y_train)

        return mlflow.active_run().info.run_id


# COMMAND ----------
# Split data into train/val/test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    nyc_taxi_pdf, target_column="fare_amount", split_ratio=(0.7, 0.2, 0.1)
)
# COMMAND ----------
# Trigger model training
run_id = train_model(X_train, y_train)

# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model", name=REGISTERED_MODEL_NAME
)

# COMMAND ----------
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME, alias="Champion", version=model_version.version
)

# COMMAND ----------
import mlflow.pyfunc

model_uri = f"models:/{REGISTERED_MODEL_NAME}@Champion"
print(f"model_uri: {model_uri}")

# COMMAND ----------
champion_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
print(champion_model.predict(X_test)[:100])
