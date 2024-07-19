import pandas as pd
import numpy as np

# Load the data from the Excel file
file_path = 'Outfitter_Bahawalpur.xlsx'  # Update with the actual file path
data = pd.read_excel(file_path, skiprows=2, usecols='F')  # Adjust 'usecols' as needed

# Assuming the column name is automatically taken as the first row's header
column_name = data.columns[0]

# Create a copy of the original data
data['Original'] = data[column_name]

# Replace outliers and zeros
def handle_zeros_and_outliers(series, threshold=100):
    series_copy = series.copy()
    length = len(series)
    
    # Identify outliers based on threshold
    outliers = (series.diff().abs() > threshold).astype(int)
    
    for i in range(length):
        if series[i] == 0 or outliers[i] == 1:
            if i == 0:
                next_non_zero = series[i+1]
                series_copy[i] = next_non_zero
            elif i == length - 1:
                prev_non_zero = series[i-1]
                series_copy[i] = prev_non_zero
            else:
                prev_non_zero = series_copy[i-1]
                next_non_zero_index = i + 1
                while next_non_zero_index < length and (series[next_non_zero_index] == 0 or outliers[next_non_zero_index] == 1):
                    next_non_zero_index += 1
                if next_non_zero_index < length:
                    next_non_zero = series[next_non_zero_index]
                    diff = next_non_zero - prev_non_zero
                    num_zeros = next_non_zero_index - i + 1
                    step = diff / num_zeros
                    for j in range(i, next_non_zero_index):
                        series_copy[j] = prev_non_zero + (j - i + 1) * step
    return series_copy

# Apply the function to handle zeros and outliers
data[column_name] = handle_zeros_and_outliers(data[column_name])

# Save the corrected data to a new Excel file
corrected_file_path = 'corr_data.xlsx'
data.to_excel(corrected_file_path, index=False)

print(f"Corrected data saved to {corrected_file_path}")
