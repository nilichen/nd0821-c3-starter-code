import os


DATA_PATH = os.getcwd() + "/data/census_clean.csv"
DATA_SAMPLE_PATH = os.getcwd() + "/data/census_sample.csv"

MODEL_PATH = os.getcwd() + "/model/random_forest.pkl"
ONEHOT_ENCODER_PATH = os.getcwd() + "/model/onehot_encoder.pkl"
LABEL_ENCODER_PATH = os.getcwd() + "/model/label_encoder.pkl"

EXAMPLE_CENSUS_DATA_POS = {'age': 41,
                           'workclass': 'Private',
                           'fnlgt': 58880,
                           'education': 'Bachelors',
                           'education-num': 13,
                           'marital-status': 'Married-civ-spouse',
                           'occupation': 'Prof-specialty',
                           'relationship': 'Wife',
                           'race': 'White',
                           'sex': 'Female',
                           'capital-gain': 7688,
                           'capital-loss': 0,
                           'hours-per-week': 10,
                           'native-country': 'United-States'}
EXAMPLE_CENSUS_DATA_NEG = {'age': 39,
                           'workclass': 'State-gov',
                           'fnlgt': 77516,
                           'education': 'Bachelors',
                           'education-num': 13,
                           'marital-status': 'Never-married',
                           'occupation': 'Adm-clerical',
                           'relationship': 'Not-in-family',
                           'race': 'White',
                           'sex': 'Male',
                           'capital-gain': 2174,
                           'capital-loss': 0,
                           'hours-per-week': 40,
                           'native-country': 'United-States'}

ALL_FEATURES = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
                       'marital-status', 'occupation', 'relationship', 'race', 'sex',
                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                       'salary']
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = 'salary'
