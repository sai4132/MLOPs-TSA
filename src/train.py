import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv(r'D:\MLOPS\data\raw\AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df["t"] = np.arange(len(df))

# Experiment configurations
window_sizes = [12, 24, 36]  # months

for window in window_sizes:
    with mlflow.start_run():
        # Define train/test split based on window
        train = df.iloc[:-window]
        test = df.iloc[-window:]

        X_train = train[["t"]]
        y_train = train["Passengers"]

        X_test = test[["t"]]
        y_test = test["Passengers"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log experiment details
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("forecast_window", window)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Window: {window} | RMSE: {rmse}")
