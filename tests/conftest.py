import pytest

from ml import DATA_PATH, DATA_SAMPLE_PATH, MODEL_PATH, ONEHOT_ENCODER_PATH, LABEL_ENCODER_PATH
from ml.data import read_data

import pickle


@pytest.fixture(scope='session')
def data():
    return read_data(DATA_PATH)


@pytest.fixture(scope='session')
def data_sample():
    return read_data(DATA_SAMPLE_PATH)


@pytest.fixture(scope='session')
def models():
    model = pickle.load(open(MODEL_PATH, 'rb'))
    encoder = pickle.load(open(ONEHOT_ENCODER_PATH, 'rb'))
    lb = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))

    return model, encoder, lb
