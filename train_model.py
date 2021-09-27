from sklearn.model_selection import train_test_split

from ml.data import read_data, process_data
from ml.model import train_model, compute_model_metrics, compute_model_metrics_on_slices, inference
from ml import DATA_PATH, DATA_SAMPLE_PATH, CAT_FEATURES, LABEL

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


data = read_data(DATA_PATH)
train, test = train_test_split(data, test_size=0.20, random_state=0)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=CAT_FEATURES, label=LABEL, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=CAT_FEATURES, label=LABEL, training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
precision, recall, fbeta = compute_model_metrics(
    y_test, inference(model, X_test))
logger.info(
    f"Precision: {precision:.2f}, Recall: {recall:.2f}, FBeta: {fbeta:.2f}")
compute_model_metrics_on_slices(test, 'sex')
