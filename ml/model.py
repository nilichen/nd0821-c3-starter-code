
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from ml import MODEL_PATH, ONEHOT_ENCODER_PATH, LABEL_ENCODER_PATH, CAT_FEATURES
from ml.data import process_data

import pickle
import logging
import sys
from collections import namedtuple

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
    cv_rfc.fit(X_train, y_train)

    pickle.dump(cv_rfc.best_estimator_, open(MODEL_PATH, 'wb'))
    return cv_rfc.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    namedtuple('Metrics', "Precision Recall FBeta")
    """
    Metrics = namedtuple('Metrics', "Precision Recall FBeta")

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return Metrics(precision, recall, fbeta)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_model_metrics_on_slices(data, cat_feature):
    """ Outputs the performance of the model on slices of the data.

    Inputs
    ------
    data : pd.DataFrame
        Pandas dataframe to compute the metrics upon.
    cat_feature : str
        Feature to slice upon.
    Returns
    -------
    pd.DataFrame: index is cat_feature values with Precision, Recall, FBeta as columns
    """
    def compute_model_metrics_on_slice(data_slice):
        X_slice, y_slice, _, _ = process_data(
            data_slice, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder, lb=lb)
        return compute_model_metrics(y_slice, inference(model, X_slice))

    logger.info("-" * 50)
    model = pickle.load(open(MODEL_PATH, 'rb'))
    encoder = pickle.load(open(ONEHOT_ENCODER_PATH, 'rb'))
    lb = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))

    metrics_on_slices = pd.DataFrame.from_dict(
        data.groupby([cat_feature]).apply(
            compute_model_metrics_on_slice
        ).to_dict(), orient='index').round(2)

    logger.info('\t' + metrics_on_slices.to_string().replace('\n', '\n\t'))
    return metrics_on_slices
