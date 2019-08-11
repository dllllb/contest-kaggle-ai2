import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.joblib import Parallel


class ModelEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, assembly_estimator, intermediate_estimators, ensemble_train_size=.25, n_jobs=1):
        self.assembly_estimator = assembly_estimator
        self.intermediate_estimators = intermediate_estimators
        self.ensemble_train_size = ensemble_train_size
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split

        if self.ensemble_train_size == 1:
            X_train1, y_train1 = X, y
            X_train2, y_train2 = X, y
        else:
            X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y, test_size=self.ensemble_train_size)

        Parallel(n_jobs=self.n_jobs)(
            ((fit_est, [est, X_train1, y_train1], {}) for est in self.intermediate_estimators)
        )

        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X_train2], {}) for est in self.intermediate_estimators)
        ))

        self.assembly_estimator.fit(probas, y_train2)

        return self

    def predict(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_estimators)
        ))
        return self.assembly_estimator.predict(probas)

    def predict_proba(self, X):
        probas = np.hstack(Parallel(n_jobs=self.n_jobs)(
            ((predict_proba_est, [est, X], {}) for est in self.intermediate_estimators)
        ))
        return self.assembly_estimator.predict_proba(probas)


def fit_est(estimator, features, labels):
    return estimator.fit(features, labels)


def predict_proba_est(estimator, features):
    return estimator.predict_proba(features)
