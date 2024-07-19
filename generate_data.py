import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt



def generate_data(n=100, seed=42):
    """
    Generate synthetic time series data with exogenous variables.
    
    Parameters:
    n (int): Number of data points.
    seed (int): Seed for reproducibility.
    
    Returns:
    pd.DataFrame: DataFrame containing the endogenous and exogenous variables.
    """
    np.random.seed(seed)
    Y = np.cumsum(np.random.randn(n))  # Random walk
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2}, index=pd.date_range(start='2020-01-01', periods=n))
    return data

def fit_arimax(data, order=(1, 1, 1), exog_cols=['X1', 'X2'], forecast_steps=10):
    """
    Fit an ARIMAX model to the data and make forecasts.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the time series and exogenous variables.
    order (tuple): The (p,d,q) order of the ARIMA model.
    exog_cols (list): List of column names for the exogenous variables.
    forecast_steps (int): Number of steps to forecast.
    
    Returns:
    pd.DataFrame: DataFrame containing the observed, forecasted values, and confidence intervals.
    """
    exog = data[exog_cols]
    model = SARIMAX(data['Y'], exog=exog, order=order)
    results = model.fit()
    print(results.summary())

    forecast = results.get_forecast(steps=forecast_steps, exog=np.random.randn(forecast_steps, len(exog_cols)))
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    forecast_df = pd.DataFrame({'Forecast': forecast_mean}, index=forecast_index)
    forecast_df['Lower CI'] = forecast_ci.iloc[:, 0]
    forecast_df['Upper CI'] = forecast_ci.iloc[:, 1]
    
    return forecast_df

def plot_results(data, forecast_df):
    """
    Plot the observed data and forecasted values.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing the observed time series.
    forecast_df (pd.DataFrame): DataFrame containing the forecasted values and confidence intervals.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Y'], label='Observed')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast')
    plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.3)
    plt.legend()
    plt.title('ARIMAX Model Forecast')
    plt.show()

# Generate synthetic data
data = generate_data()
# Fit ARIMAX model and forecast
forecast_df = fit_arimax(data, order=(1, 1, 1), exog_cols=['X1', 'X2'], forecast_steps=10)
# Plot the results
plot_results(data, forecast_df)
