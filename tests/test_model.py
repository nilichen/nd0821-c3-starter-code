
import numpy as np
import pandas as pd

from ml import ALL_FEATURES, CAT_FEATURES, LABEL, EXAMPLE_CENSUS_DATA_POS, EXAMPLE_CENSUS_DATA_NEG
from ml.data import process_data
from ml.model import inference, compute_model_metrics, compute_model_metrics_on_slices

THRESHOLD = 0.5


def test_column_names(data):
    assert list(ALL_FEATURES) == list(data.columns.values)


def test_model_inference(models):
    model, encoder, _ = models
    X, _, _, _ = process_data(pd.DataFrame(
        [EXAMPLE_CENSUS_DATA_POS, EXAMPLE_CENSUS_DATA_NEG]), categorical_features=CAT_FEATURES, training=False, encoder=encoder)
    print(inference(model, X))
    assert np.array_equal(inference(model, X), np.array([1, 0]))


def test_model_performance(data_sample, models):
    model, encoder, lb = models
    X, y, _, _ = process_data(
        data_sample, categorical_features=CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb)
    print(X)
    precision, recall, fbeta = compute_model_metrics(y, inference(model, X))

    assert precision > THRESHOLD, "Precision is above THRESHOLD: {THRESHOLD}"
    assert recall > THRESHOLD, "Recall is above THRESHOLD: {THRESHOLD}"
    assert fbeta > THRESHOLD, "FBeta is above THRESHOLD: {THRESHOLD}"


def test_model_gender_bias(data_sample):
    gender_df = compute_model_metrics_on_slices(data_sample, 'sex')

    assert (gender_df.loc['Female'] - gender_df.loc['Male']
            ).abs().mean() < 0.065, "Performace differs between female and male"
