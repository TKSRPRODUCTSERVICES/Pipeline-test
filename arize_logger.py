import os
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments

# Function to log data to Arize
def log_to_arize(x_test, y_pred, y_test, config):
    x_test = x_test.copy()
    x_test["prediction"] = y_pred  
    x_test["actual"] = y_test.values  
    x_test["id"] = range(len(x_test))  
    x_test.reset_index(drop=True, inplace=True)

    # Define Feature Columns
    feature_columns = [col for col in x_test.columns if col not in ["prediction", "actual", "id"]]

    # Load Arize API Credentials
    API_KEY = "d33ea83d527145c5ae0"
    SPACE_ID = "U3BhY2U6MTUxMDI6dURzQw=="
    arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)


    # Define Schema
    schema = Schema(
        prediction_id_column_name="id",
        prediction_label_column_name="prediction",
        actual_label_column_name="actual",
        feature_column_names=feature_columns
    )

    # Log Data to Arize
    response = arize_client.log(
        dataframe=x_test,
        schema=schema,
        model_id=config["pipeline"]["steps"]["log_to_arize"]["arize"]["model_id"],
        model_version=config["pipeline"]["steps"]["log_to_arize"]["arize"]["model_version"],
        model_type=ModelTypes.REGRESSION,
        environment=Environments.PRODUCTION
    )

    print(f"Arize Response: {response}")
