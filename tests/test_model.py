def test_column_names(data):

    expected_colums = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
                       'marital-status', 'occupation', 'relationship', 'race', 'sex',
                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                       'salary']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)
