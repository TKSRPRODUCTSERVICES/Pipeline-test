import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Configuration
def load_config(config_path="pipeline_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load Dataset
def load_data(config):
    file_path = config["pipeline"]["steps"]["load_data"]["file_path"]
    df = pd.read_csv(file_path)
    
    # Handle missing values
    drop_na = config["pipeline"]["steps"]["preprocessing"]["drop_na"]
    data = df.dropna().copy() if drop_na else df.copy()
    return data

# Preprocess Data (Handle Categorical Encoding)
def preprocess_data(data):
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    return data
