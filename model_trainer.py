import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train and Evaluate Model
def train_model(data, config):
    test_size = config["pipeline"]["parameters"]["test_size"]
    random_state = config["pipeline"]["parameters"]["random_state"]
    
    # Define Features & Target
    x = data.drop(columns="World Population Percentage")
    y = data["World Population Percentage"]

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Train Model
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return model, x_test, y_test, y_pred
