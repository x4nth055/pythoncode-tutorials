import pandas as pd

# Config settings
pd.set_option('max_columns', None)
pd.set_option('max_rows', 12)

# Import CSV data
data_frames = pd.read_csv (r'simulated_data.csv')

# Data Type Conversion
# Remove '$' from donation strings
data_frames['donation'] = data_frames['donation'].str.strip('$')

# Convert donation stings into numerical data type
data_frames['donation'] = data_frames['donation'].astype('float64')

print(data_frames.head(10))
print(data_frames.info())