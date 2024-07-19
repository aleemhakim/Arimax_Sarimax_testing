
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the data from the Excel file
file_path = 'Outfitter_Bahawalpur.xlsx'  # Update with the actual file path
data = pd.read_excel(file_path, skiprows=2, usecols='F')  # Adjust 'usecols' as needed

# Assuming the column name is automatically taken as the first row's header
column_name = data.columns[0]

# Replace 0.000 with NaN
data.replace(0.000, np.nan, inplace=True)

# Identify and handle outliers
# Assuming outliers are values that deviate significantly from the previous values
threshold = 100  # Define a threshold for outlier detection

def detect_outliers(series):
    return (series.diff().abs() > threshold).astype(int)

outliers = detect_outliers(data[column_name])

# Fit the SARIMAX model
# Adjust the parameters (order, seasonal_order) based on your data characteristics
sarimax_model = SARIMAX(data[column_name], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
sarimax_results = sarimax_model.fit(disp=False)

# Fill missing values using the SARIMAX model
data_filled = data.copy()
data_filled.loc[data_filled[column_name].isna(), column_name] = sarimax_results.predict(start=data_filled.index.min(), end=data_filled.index.max())[data_filled[column_name].isna()]

# Correct outliers using the SARIMAX model
data_filled.loc[outliers == 1, column_name] = sarimax_results.predict(start=data_filled.index.min(), end=data_filled.index.max())[outliers == 1]

# Function to iteratively adjust values
def adjust_values(series):
    corrected_series = series.copy()
    for i in range(1, len(series) - 1):
        if series[i] < series[i - 1]:
            corrected_series[i] = series[i - 1]
        if series[i] > series[i + 1]:
            corrected_series[i] = series[i + 1]
    return corrected_series

# Apply the adjustment function iteratively
for _ in range(10):  # Iterate multiple times to ensure values are adjusted correctly
    data_filled[column_name] = adjust_values(data_filled[column_name])


# Save the corrected data to a new Excel file
corrected_file_path = 'corrected_data1.xlsx'
data_filled.to_excel(corrected_file_path, index=False)

print(f"Corrected data saved to {corrected_file_path}")
