import pandas as pd
import sklearn.preprocessing as sklp

# Importing Beach Dataset

weatherdata = pd.read_csv('weatherdata.csv', sep=',')
weatherdata.columns = ['desc', 'temperature', 'pressure', 'humidity', 'wind_str', 'wind_deg', 'beachday?']

# Dropping Lines With Missing Values

weatherdata = weatherdata.dropna()


# Encoding Categorical Values

weatherEncoded = weatherdata.copy()
encoder = sklp.LabelEncoder()
weatherEncoded['desc'] = (encoder.fit_transform(weatherEncoded['desc']))
weatherEncoded['beachday?'] = weatherEncoded['beachday?'].astype(int)