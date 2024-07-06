import os.path
import joblib
from datetime import datetime

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import DateTimeTransformer, AirportLatLongTransformer

df = pd.read_csv('data/2019_prepared.csv')

def save_model(model):
    with open('models/rfc.pkl', 'wb') as f:
        joblib.dump(model, f)


def load_model():
    with open('models/rfc.pkl', 'rb') as f:
        model = joblib.load(f)
    return model


def create_model():
    y = df['DELAY_CATEGORY']
    X = df.drop(columns='DELAY_CATEGORY')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = ['FL_DAY', 'DEP_MINUTES', 'DAY_OF_WEEK', 'ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT', 'DEST_LON']
    categorical_features = ['OP_UNIQUE_CARRIER']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numerical_features),
        ]
    )

    calibrated_clf = CalibratedClassifierCV(
        RandomForestClassifier(max_depth=30, min_samples_leaf=4, min_samples_split=10, n_estimators=300), cv=5,
        method="isotonic")

    clf = Pipeline([
        ('datetime_transformer', DateTimeTransformer()),
        ('airport_transformer', AirportLatLongTransformer()),
        ('preprocessor', preprocessor),
        ('classifier', calibrated_clf)
    ])

    clf.fit(X_train, y_train)

    save_model(clf)
    return clf


if os.path.exists('models/rfc.pkl'):
    print('\nLoading predictive model...')
    model = load_model()
else:
    print('\nNo saved model found. Creating new predictive model...')
    model = create_model()

print('Predictive model loaded.')

airlines_df = pd.read_csv('data/airlines_data.csv')

airports_df = pd.read_csv('data/airports.csv')
airports_df['Code'] = airports_df['iata']
airports_df['Airport Name'] = airports_df['airport']
airports_df = airports_df[['Code', 'Airport Name']]

valid_airports = pd.concat([df['ORIGIN'], df['DEST']]).unique()
airports_df = airports_df[airports_df['Code'].isin(valid_airports)]


def airport_search():
    search_term = input("\nEnter part of the airport or city name or type 'cancel': ").strip()
    if search_term.lower() == 'cancel':
        return
    result = airports_df[airports_df['Airport Name'].str.contains(search_term, case=False, na=False)]

    if not result.empty:
        print(f"\n{result}\n")
    else:
        response = input(f"\nCould not find an airport matching that search term. Try again (Y/n)? ")
        if response.lower() == 'n':
            return
        else:
            return airport_search()


def start_repl():
    user_input = {
        'FL_DATE': None,
        'OP_UNIQUE_CARRIER': None,
        'ORIGIN': None,
        'DEST': None,
        'DEP_TIME': None
    }

    user_input_complete = [False]

    def prompt_user():
        if user_input['FL_DATE'] is None:
            fl_date = input("\nEnter the flight date using format YYYY-MM-DD: ")
            try:
                datetime.strptime(fl_date, '%Y-%m-%d')
                user_input['FL_DATE'] = fl_date
            except ValueError:
                print("Incorrect date format.")
        elif user_input['DEP_TIME'] is None:
            dep_time = input("Enter the (local) departure time using format HH:MM (24-hour clock): ")
            try:
                datetime.strptime(dep_time, '%H:%M')
                hours, minutes = dep_time.split(':')
                user_input['DEP_TIME'] = float(hours) * 100 + float(minutes)
            except ValueError:
                print("Incorrect time format.")
        elif user_input['OP_UNIQUE_CARRIER'] is None:
            carrier = input(
                "Enter the 2-letter airline identifier (i.e. DL for Delta Airlines, UA for United Airlines) or enter 'list' for a list of airlines: ").upper()
            if carrier == 'LIST':
                print(f"\n{airlines_df}\n")
            elif len(carrier) != 2:
                print("Incorrect 2-letter airline identifier.")
            else:
                user_input['OP_UNIQUE_CARRIER'] = carrier
        elif user_input['ORIGIN'] is None:
            origin = input(
                "Enter the origin 3-letter airport code (i.e. ATL for Atlanta, LAX for Los Angeles) or enter 'search' to search by airport name: ").strip().upper()
            if origin == 'SEARCH':
                airport_search()
            elif origin not in airports_df['Code'].values:
                print("Incorrect 3-letter airport code.")
            else:
                user_input['ORIGIN'] = origin
        elif user_input['DEST'] is None:
            dest = input(
                "Enter the destination 3-letter airport code (i.e. ATL for Atlanta, LAX for Los Angeles). Enter 'search' to search by airport name: ").strip().upper()
            if dest == 'SEARCH':
                airport_search()
            elif dest not in airports_df['Code'].values:
                print("Incorrect 3-letter airport code.")
            else:
                user_input['DEST'] = dest
        else:
            user_input_complete[0] = True

    while not user_input_complete[0]:
        prompt_user()

    user_df = pd.DataFrame([user_input])

    user_pred_proba = model.predict_proba(user_df)
    class_labels = model.named_steps['classifier'].classes_
    proba_map = {class_label: proba for class_label, proba in zip(class_labels, user_pred_proba[0])}

    chance_on_time = proba_map['NO_DELAY'] * 100
    chance_delayed = 100 - chance_on_time

    print(f"\nOn-time arrival probability: {chance_on_time:.2f}%")
    print(f"Delayed arrival probability: {chance_delayed:.2f}%")

    print(f"\nChance of minor delay (15min-45min): {proba_map['MINOR_DELAY'] * 100:.2f}%")
    print(f"Chance of major delay (45min-2hrs):  {proba_map['MAJOR_DELAY'] * 100:.2f}%")
    print(f"Chance of severe delay (>2hrs):  {proba_map['SEVERE_DELAY'] * 100:.2f}%")

    again = input("\nCheck another flight (y/N)? ")

    if again.lower() == 'y':
        start_repl()


start_repl()
