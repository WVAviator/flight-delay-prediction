import pandas as pd

from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Converts a column of name 'FL_DATE' into three columns, 'FL_DAY', 'DEP_MINUTES', and 'DAY_OF_WEEK' These
    represent the days since January 1st, the minutes since midnight, and the day of the week (0-6, Monday-Sunday),
    respectively. The 'FL_DATE' column is removed.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['FL_DAY'] = X['FL_DATE'].apply(self.days_since_year_start)
        X_transformed['DEP_MINUTES'] = X['DEP_TIME'].apply(self.minutes_since_midnight)
        X_transformed['DAY_OF_WEEK'] = X['FL_DATE'].apply(self.day_of_week)
        return X_transformed.drop(columns=['FL_DATE', 'DEP_TIME'])

    def days_since_year_start(self, date_str: str) -> int:
        input_date = datetime.strptime(date_str, "%Y-%m-%d")
        january_first = datetime(input_date.year, 1, 1)
        days_difference = (input_date - january_first).days
        return days_difference

    def minutes_since_midnight(self, time_float: float) -> int:
        hours = int(time_float // 100)
        minutes = int(time_float % 100)
        total_minutes = hours * 60 + minutes
        return total_minutes

    def day_of_week(self, date_str: str) -> int:
        input_date = datetime.strptime(date_str, "%Y-%m-%d")
        return input_date.weekday()


class AirportLatLongTransformer(BaseEstimator, TransformerMixin):
    """
    Takes the value of columns 'ORIGIN' and 'DEST' and performs a lookup and adds additional columns 'ORIGIN_LAT',
    'ORIGIN_LONG', 'DEST_LAT' and 'DEST_LONG', representing the latitude and longitude values for the provided
    airport code.
    """
    def __init__(self):
        airports_df = pd.read_csv('data/airports.csv')
        self.airports_dict = airports_df[['iata', 'latitude', 'longitude']].set_index('iata').T.to_dict('list')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in ['ORIGIN', 'DEST']:
            lat_col = col + '_LAT'
            lon_col = col + '_LON'
            X_transformed[lat_col] = X[col].apply(lambda x: self.airports_dict[x][0])
            X_transformed[lon_col] = X[col].apply(lambda x: self.airports_dict[x][1])
        return X_transformed
