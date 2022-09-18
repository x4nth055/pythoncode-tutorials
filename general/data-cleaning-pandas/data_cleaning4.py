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


# Handle Data Inconsistencies
# Capitalize strings
data_frames['street_address'] = data_frames['street_address'].str.split()

def capitalize_words(arr):
    for index, word in enumerate(arr):
        if index == 0:
            pass
        else:
            arr[index] = word.capitalize()

data_frames['street_address'].apply(lambda x: capitalize_words(x))
data_frames['street_address'] = data_frames['street_address'].str.join(' ')

print(data_frames['street_address'])
