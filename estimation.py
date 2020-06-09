from hyperopt import hp, fmin, tpe, STATUS_OK
from lightgbm import LGBMClassifier
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC, OneClassSVM

import utilities as ut


class BasicEstimator:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.params = {}
        self.estimator = None

    def basic_pipeline(self):
        pipe = pipeline.Pipeline(steps=[("model_fitting", self.estimator)])
        return pipe


class SupervisedEstimator(BasicEstimator):
    def __init__(self, features, labels):
        BasicEstimator.__init__(self, features, labels)
        self.grid = None

    def supervised_pipeline(self):
        train_features, test_features, train_labels, test_labels = model_selection.train_test_split(self.features,
                                                                                                    self.labels,
                                                                                                    test_size=0.2,
                                                                                                    stratify=self.labels,
                                                                                                    shuffle=True)
        self.grid.fit(train_features, train_labels)
        prediction = self.grid.best_estimator_.predict_proba(test_features)[:, 1]
        roc_auc = metrics.roc_auc_score(test_labels, prediction)
        pr_curve = metrics.precision_recall_curve(test_labels, prediction)
        pr_auc = metrics.auc(pr_curve[1], pr_curve[0])
        f_measure = metrics.f1_score(test_labels, self.grid.best_estimator_.predict(test_features))
        roc_curve = metrics.roc_curve(test_labels, prediction)
        full_prediction = self.grid.best_estimator_.predict(self.features)
        ut.plot_results(roc_curve, pr_curve)
        low_dim = ut.calculate_pca(self.features)
        ut.plot_low_dimensions(low_dim, full_prediction)
        return roc_auc, pr_auc, f_measure


class KNNEstimator(SupervisedEstimator):
    def __init__(self, features, target):
        SupervisedEstimator.__init__(self, features, target)
        self.estimator = KNeighborsClassifier(n_jobs=-1)
        self.params = {"model_fitting__n_neighbors": [1, 2, 3, 5, 10],
                       "model_fitting__p": [1, 2],
                       "model_fitting__weights": ["uniform", "distance"]}
        self.grid = GridSearchCV(self.basic_pipeline(), self.params, scoring="roc_auc", cv=10)


class LogisticRegressionEstimator(SupervisedEstimator):
    def __init__(self, features, labels):
        SupervisedEstimator.__init__(self, features, labels)
        self.estimator = LogisticRegression(n_jobs=-1, max_iter=1e5,
                                            solver="saga", penalty="elasticnet",
                                            l1_ratio=0.1)
        self.params = {"model_fitting__l1_ratio": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]}
        self.grid = GridSearchCV(self.basic_pipeline(), self.params, scoring="roc_auc", cv=10)


class SVMEstimator(SupervisedEstimator):
    def __init__(self, features, labels):
        SupervisedEstimator.__init__(self, features, labels)
        self.estimator = SVC(kernel="rbf", cache_size=2048, probability=True)
        self.params = {"model_fitting__C": [0.001, 0.1, 1, 10]}
        self.grid = GridSearchCV(self.basic_pipeline(), self.params, scoring="roc_auc", cv=5)


class BoostingMachineEstimator(SupervisedEstimator):
    def __init__(self, features, labels):
        SupervisedEstimator.__init__(self, features, labels)
        self.estimator = LGBMClassifier(n_jobs=-1, n_estimators=1000, objective='binary')
        self.params = {"model_fitting__learning_rate ": [0.1, 0.01],
                       "model_fitting__reg_alpha": [0.1, 0.01, 0.001]}
        self.grid = GridSearchCV(self.basic_pipeline(), self.params, scoring="roc_auc", cv=5)


class UnsupervisedEstimator(BasicEstimator):
    def __init__(self, features, labels):
        BasicEstimator.__init__(self, features, labels)
        self.space = None

    def get_best(self):
        pass

    def basic_pipeline(self):
        return self.estimator

    def unsupervised_estimation(self):
        best_estimator = self.estimator(**self.get_best())
        prediction = best_estimator.fit_predict(self.features)
        ari = metrics.adjusted_rand_score(self.labels, prediction)
        mi = metrics.mutual_info_score(self.labels, prediction)
        ami = metrics.adjusted_mutual_info_score(self.labels, prediction)
        nmi = metrics.normalized_mutual_info_score(self.labels, prediction)
        try:
            calinski_harabasz = metrics.calinski_harabasz_score(self.features, prediction)
            silhouette = metrics.silhouette_score(self.features, prediction)
        except ValueError:
            calinski_harabasz = 0
            silhouette = 0
        low_dim = ut.calculate_pca(self.features)
        ut.plot_low_dimensions(low_dim, prediction)
        return ari, mi, ami, nmi, calinski_harabasz, silhouette


class IsolationForestEstimator(UnsupervisedEstimator):
    def __init__(self, features, labels):
        UnsupervisedEstimator.__init__(self, features, labels)
        self.estimator = IsolationForest
        self.space = {'n_estimators': hp.uniform('n_estimators', 100, 1000),
                      'max_features': hp.uniform('max_features', 0.7, 1.0),
                      }

    def get_best(self):
        def objective(space):
            params = {
                'n_estimators': int(space['n_estimators']),
                'max_features': space['max_features']
            }
            estimator = IsolationForest(n_jobs=-1, **params)
            prediction = estimator.fit_predict(self.features)
            score = -metrics.calinski_harabasz_score(self.features, prediction)
            return {'loss': score, 'status': STATUS_OK}

        best_params = fmin(fn=objective, space=self.space, algo=tpe.suggest, max_evals=10)
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['n_jobs'] = -1
        return best_params


class DBScanEstimator(UnsupervisedEstimator):
    def __init__(self, features, labels):
        UnsupervisedEstimator.__init__(self, features, labels)
        self.estimator = DBSCAN
        self.space = {'eps': hp.uniform('eps', 0.1, 0.8)}

    def get_best(self):
        def objective(space):
            params = {'eps': space['eps']}
            estimator = DBSCAN(n_jobs=-1, **params)
            prediction = estimator.fit_predict(self.features)
            score = 1 - metrics.mutual_info_score(self.labels, prediction)
            return {'loss': score, 'status': STATUS_OK}

        best_params = fmin(fn=objective, space=self.space, algo=tpe.suggest, max_evals=5)
        best_params['n_jobs'] = -1
        return best_params


class OneClassSVMEstimator(UnsupervisedEstimator):
    def __init__(self, features, labels):
        UnsupervisedEstimator.__init__(self, features, labels)
        self.estimator = OneClassSVM
        self.space = {'nu': hp.uniform('nu', 0.01, 0.5),
                      'gamma': hp.uniform('gamma', 0.01, 0.5)
                      }

    def get_best(self):
        def objective(space):
            params = {'nu': space['nu'], 'gamma': space['gamma']}
            estimator = OneClassSVM(cache_size=2048, kernel='rbf', **params)
            prediction = estimator.fit_predict(self.features)
            score = -metrics.calinski_harabasz_score(self.features, prediction)
            return {'loss': score, 'status': STATUS_OK}

        best_params = fmin(fn=objective, space=self.space, algo=tpe.suggest, max_evals=5)
        best_params['cache_size'] = 2048
        best_params['kernel'] = 'rbf'
        return best_params


class LocalOutlierFactorEstimator(UnsupervisedEstimator):
    def __init__(self, features, labels):
        UnsupervisedEstimator.__init__(self, features, labels)
        self.estimator = LocalOutlierFactor
        self.space = {'n_neighbors': hp.uniform('n_neighbors', 10, 50),
                      'leaf_size': hp.uniform('leaf_size', 30, 300)
                      }

    def get_best(self):
        def objective(space):
            params = {'n_neighbors': int(space['n_neighbors']),
                      'leaf_size': int(space['leaf_size'])
                      }
            estimator = LocalOutlierFactor(n_jobs=-1, **params)
            prediction = estimator.fit_predict(self.features)
            score = -metrics.calinski_harabasz_score(self.features, prediction)
            return {'loss': score, 'status': STATUS_OK}

        best_params = fmin(fn=objective, space=self.space, algo=tpe.suggest, max_evals=5)
        best_params['n_neighbors'] = int(best_params['n_neighbors'])
        best_params['leaf_size'] = int(best_params['leaf_size'])
        best_params['n_jobs'] = -1
        return best_params
