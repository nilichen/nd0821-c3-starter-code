from ml import ALL_FEATURES


def test_column_names(data):
    assert list(ALL_FEATURES) == list(data.columns.values)
