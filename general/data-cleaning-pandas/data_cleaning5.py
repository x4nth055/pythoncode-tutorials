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
# Normalize strings
data_frames['street_address'] = data_frames['street_address'].str.split()

def normalize_words(arr):
    for index, word in enumerate(arr):
        if index == 0:
            pass
        else:
            arr[index] = normalize(word)

def normalize(word):
    if word.lower() == 'st':
        word = 'street'
    elif word.lower() == 'rd':
        word = 'road'

    return word.capitalize()


data_frames['street_address'].apply(lambda x: normalize_words(x))
data_frames['street_address'] = data_frames['street_address'].str.join(' ')

print(data_frames.head(10))