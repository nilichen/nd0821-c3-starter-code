# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Date: 2021 Sep 26
- Type: Random Forest
- Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Performance: Precision: 0.73, Recall: 0.63, FBeta: 0.67

Tuned hyperparameter on 5-fold cv to get the best candidate

## Intended Use

Predict whether income exceeds $50K/yr based on census data.

## Training Data

https://archive.ics.uci.edu/ml/datasets/census+income

- Proprocess: remove the spaces

## Evaluation Data

20% of the training data is used to evaluate the model

## Metrics

Precision, Recall, FBeta

## Ethical Considerations

None

## Caveats and Recommendations

None
