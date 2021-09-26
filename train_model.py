# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import read_data, process_data
from ml.model import train_model, compute_model_metrics, inference
from ml import CAT_FEATURES, LABEL


data = read_data()
train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label=LABEL, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
compute_model_metrics(y_test, inference(model, X_test))
