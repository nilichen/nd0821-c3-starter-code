import pytest

from ml.data import read_data


@pytest.fixture(scope='session')
def data():
    return read_data()
