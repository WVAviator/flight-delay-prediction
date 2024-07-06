import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from transformers import DateTimeTransformer, AirportLatLongTransformer

def save_model(model):
def create_model():
    df = pd.read_csv('data/2019_prepared.csv')

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

    gauge_performance(clf, X_test, y_test)
    return clf


def gauge_performance(clf, X_test, y_test):
    y_pred_proba = clf.predict_proba(X_test)

    classes = clf.named_steps['classifier'].classes_
    brier_scores = []

    for i, class_label in enumerate(classes):
        brier_score = brier_score_loss(y_test == class_label, y_pred_proba[:, i])
        brier_scores.append(brier_score)
        print(f'Brier score for class {class_label}: {brier_score}')

    average_brier_score = np.mean(brier_scores)
    print(f'Average Brier score: {average_brier_score}')

    print("\nLog loss (smaller is better):")
    print(log_loss(y_test, y_pred_proba))

create_model()