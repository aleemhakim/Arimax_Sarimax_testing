import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = 'Outfitter_Bahawalpur.xlsx'  # Update with the actual file path
data = pd.read_excel(file_path, skiprows=2, usecols='F')  # Adjust 'usecols' as needed

# Replace 0.000 with NaN
data.replace(0.000, np.nan, inplace=True)

# Identify and handle outliers
# Assuming outliers are values that deviate significantly from the previous values
threshold = 100  # Define a threshold for outlier detection

def detect_outliers(series):
    return (series.diff().abs() > threshold).astype(int)

outliers = detect_outliers(data['ColumnName'])  # Replace 'ColumnName' with the actual column name

# Fit the SARIMAX model
# Adjust the parameters (order, seasonal_order) based on your data characteristics
sarimax_model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
sarimax_results = sarimax_model.fit(disp=False)

# Fill missing values using the SARIMAX model
data_filled = data.copy()
data_filled.loc[data_filled.isna()] = sarimax_results.predict(start=data_filled.index.min(), end=data_filled.index.max())[data_filled.isna()]

# Correct outliers using the SARIMAX model
data_filled[outliers == 1] = sarimax_results.predict(start=data_filled.index.min(), end=data_filled.index.max())[outliers == 1]

# Save the corrected data to a new Excel file
corrected_file_path = 'corrected_data11.xlsx'
data_filled.to_excel(corrected_file_path, index=False)

print(f"Corrected data saved to {corrected_file_path}")
