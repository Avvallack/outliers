from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def basic_pipeline(classifier):
    """
    preparing basic pipeline
    :return:
    :param classifier: classifier instance
    :return: pipeline object
    """
    pipe = pipeline.Pipeline(steps=[('model_fitting', classifier)])
    return pipe


def supervised_estimation(grid, features, labels):
    """
    supervised estimation process
    estimating given grid and plotting roc curve and pr curve
    :param grid: gridsearch sklearn object
    :param features: pandas features to train model
    :param labels:  np array like labels
    :return: roc_auc, pr_auc, f1 metrics
    """
    train_features, test_features, train_labels, test_labels = model_selection.train_test_split(features, labels,
                                                                                                test_size=0.2,
                                                                                                stratify=labels,
                                                                                                shuffle=True)
    grid.fit(train_features, train_labels)
    prediction = grid.best_estimator_.predict_proba(test_features)[:, 1]
    roc_auc = metrics.roc_auc_score(test_labels, prediction)
    precision, recall, _ = metrics.precision_recall_curve(test_labels, prediction)
    pr_auc = metrics.auc(recall, precision)
    f1 = metrics.f1_score(test_labels, grid.predict(test_features))
    roc_curve = metrics.roc_curve(test_labels, prediction)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax[0].set_title('ROC Curve')
    ax[1].set_title('PR Curve')
    sns.lineplot(roc_curve[0], roc_curve[1], ax=ax[0])
    sns.lineplot(recall, precision, ax=ax[1])

    return roc_auc, pr_auc, f1


def knn():
    """
    KNN grid search implementation
    :return: grid search sklearn object
    """
    classifier = KNeighborsClassifier(n_jobs=-1)
    pipe = basic_pipeline(classifier)
    param_grid = {'model_fitting__n_neighbors': [1, 2, 3, 5, 10],
                  'model_fitting__p': [1, 2],
                  'model_fitting__weights': ['uniform', 'distance']}
    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring=['roc_auc'],
                        cv=20,
                        refit='roc_auc')
    return grid


def log_regression():
    """
    Log regression grid search implementation
    :return: grid search sklearn object
    """
    classifier = LogisticRegressionCV(n_jobs=-1, max_iter=1e5, solver='saga', scoring='roc_auc')
    pipe = basic_pipeline(classifier)
    param_grid = {'model_fitting__penalty': ['elasticnet'],
                  'model_fitting__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1]}
    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring=['roc_auc'],
                        cv=20,
                        refit='roc_auc')
    return grid


def svm_classifier():
    """
    svm classifier grid search implementation
    :return: grid search sklearn object
    """
    classifier = SVC(gamma='scale', probability=True, class_weight={0: 0.0001, 1: 1})
    pipe = basic_pipeline(classifier)
    param_grid = {'model_fitting__C': [0.001, 0.003, 0.1, 0.3, 1, 3, 10, 30]}
    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring=['roc_auc'],
                        cv=20,
                        refit='roc_auc')
    return grid


def boosting_machine():
    """
    lightgbm classifier grid search implementation
    :return: grid search sklearn object
    :return:
    """
    classifier = LGBMClassifier(n_jobs=-1, n_estimators=1e4, objective='binary', class_weights={0: 0.0001, 1: 1})
    pipe = basic_pipeline(classifier)
    param_grid = {'model_fitting__max_depth': [-1, 3, 10, 30, 100, 300],
                  'model_fitting__learning_rate ': [0.1, 1e-2, 1e-3, 1e-4],
                  'model_fitting__reg_alpha': [0.1, 1e-2, 1e-3],
                  'model_fitting__reg_lambda': [0.1, 1e-2, 1e-3]}
    grid = GridSearchCV(pipe,
                        param_grid,
                        scoring=['roc_auc'],
                        cv=20,
                        refit='roc_auc')
    return grid
