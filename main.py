from data_loader import load_config, load_data, preprocess_data
from model_trainer import train_model
from arize_logger import log_to_arize

# Load Configuration
config = load_config()

# Load and Preprocess Data
data = load_data(config)
data = preprocess_data(data)

# Train Model
model, x_test, y_test, y_pred = train_model(data, config)

# Log Results to Arize
log_to_arize(x_test, y_pred, y_test, config)
