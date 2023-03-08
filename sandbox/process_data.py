import numpy as np
import polars as pl
import pandas as pd
from enum import Enum
from typing import Any
from functools import singledispatch
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sandbox.fairness_metrics import test_independence, test_seperation, test_sufficiency

data = pl.read_csv("data/catalan-juvenile-recidivism-subset.csv")

# one-hot encode string columns
data = data.to_dummies(columns=data.select(pl.col(pl.Utf8)).columns)

# post-process feature map
Features = Enum("Features", data.columns)

train, validate, test = np.split(
    data.sample(frac=1, seed=123).to_numpy(),  # shuffle the dataset
    [int(0.6 * len(data)), int(0.8 * len(data))],  # split into 3 parts
)

X_train = train[:, data.columns != "V115_RECID2015_recid"].squeeze()
y_train = train[:, Features.V115_RECID2015_recid.value]
X_test = test[:, data.columns != "V115_RECID2015_recid"].squeeze()
y_test = test[:, Features.V115_RECID2015_recid.value]

classifier = DecisionTreeClassifier(criterion="gini", splitter="best", random_state=123)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
classifier.predict_proba

test_independence(classifier, X_test, "V1_sex_male", "V1_sex_female")
test_seperation(classifier, X_test, y_test, "V1_sex_male", "V1_sex_female")
test_sufficiency(classifier, X_test, y_test, "V1_sex_male", "V1_sex_female")

# TODO: generalize fairness evaluation
# @dataclass
# class EvaluateFairness:
#     @singledispatch
#     def __call__(self, estimator: BaseEstimator, X, *args: Any, **kwds: Any) -> Any:
#         # self.independence(X)
#         # self.separation(X)
#         self.estimator = estimator
#         self.X = X
#         # self.y = y

#     def independence(self, a, b):
#         # create two groups
#         X_a = self.X[self.X[a] == 1]
#         X_b = self.X[self.X[b] == 1]

#         # check prediction for group a and b
#         y_pred_a = self.estimator.predict(X_a)
#         y_pred_b = self.model.predict(X_b)

#         # calculate p(y_pred=1|A=a) and p(y_pred=1|A=b)
#         p_a = np.mean(y_pred_a)
#         p_b = np.mean(y_pred_b)

#         # print results
#         print("p(y_pred=1|A={}) = ".format(a), p_a)
#         print("p(y_pred=1|A={}) = ".format(b), p_b)

#         if p_a == p_b:
#             print("The model fulfills independence")
#         else:
#             print("The model does not fulfill independence")

#     def separation(self, a, b, y):
#         # create four groups
#         X_1_a = self.X[(self.X[a] == 1) & (y == 1)]
#         X_1_b = self.X[(self.X[b] == 1) & (y == 1)]
#         X_0_a = self.X[(self.X[a] == 1) & (y == 0)]
#         X_0_b = self.X[(self.X[b] == 1) & (y == 0)]

#         # check prediction all groups
#         y_pred_1_a = self.estimator.predict(X_1_a)
#         y_pred_1_b = self.estimator.predict(X_1_b)
#         y_pred_0_a = self.estimator.predict(X_0_a)
#         y_pred_0_b = self.estimator.predict(X_0_b)

#         # calculate p(y_pred=1|Y=1, A=a) and p(y_pred=1|Y=1, A=b) etc
#         p_1_a = np.mean(y_pred_1_a)
#         p_1_b = np.mean(y_pred_1_b)
#         p_0_a = np.mean(y_pred_0_a)
#         p_0_b = np.mean(y_pred_0_b)

#         # print results
#         print("p(y_pred=1|Y=1, A={}) = ".format(a), p_1_a)
#         print("p(y_pred=1|Y=1, A={}) = ".format(b), p_1_b)
#         print("p(y_pred=1|Y=0, A={}) = ".format(a), p_0_a)
#         print("p(y_pred=1|Y=0, A={}) = ".format(b), p_0_b)

#         if (p_1_a == p_1_b) & (p_0_a == p_0_b):
#             print("The model fulfills seperation")
#         else:
#             print("The model does not fulfill seperation")

# EvaluateFairness(classifier, X_test)
