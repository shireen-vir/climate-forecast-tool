class ClimateForecastTool:
    """
    A data science tool for climate forecasting.

    This class provides methods for data processing, model training, and prediction.
    """

    def __init__(self):
        pass


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def main():
    # Load the dataset
    data = pd.read_csv('climate_data.csv')

    # Preprocess the data
    X = data.drop(['temperature'], axis=1)
    y = data['temperature']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')


if __name__ == '__main__':
    main()