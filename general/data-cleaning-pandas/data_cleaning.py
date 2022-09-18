import pandas as pd

# Config settings
pd.set_option('max_columns', None)
pd.set_option('max_rows', 12)

# Import CSV data
data_frames = pd.read_csv (r'simulated_data.csv')

print(data_frames.head(10))
