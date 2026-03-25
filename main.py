class ClimateForecastTool:
    """
    A data science tool for climate forecasting.

    This tool uses machine learning models to predict climate patterns and trends.
    It takes in historical climate data and returns a forecast for a specified time period.
    """

    def __init__(self):
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error

        self.pd = pd
        self.np = np
        self.train_test_split = train_test_split
        self.RandomForestRegressor = RandomForestRegressor
        self.mean_squared_error = mean_squared_error

    def load_data(self, filename):
        return self.pd.read_csv(filename)

    def train_model(self, data):
        X = data.drop(['target'], axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size=0.2)
        model = self.RandomForestRegressor()
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return self.mean_squared_error(y_test, y_pred)

def main():
    tool = ClimateForecastTool()
    data = tool.load_data('climate_data.csv')
    model, X_test, y_test = tool.train_model(data)
    mse = tool.evaluate_model(model, X_test, y_test)
    print(f'Mean squared error: {mse}')

if __name__ == '__main__':
    main()