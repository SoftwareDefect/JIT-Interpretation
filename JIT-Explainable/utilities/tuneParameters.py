import optuna
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef

def tune(train_data, train_labels, test_data, test_labels, model_name):
    def objective(trial):
        if model_name == 'logistic_regression':
            C = trial.suggest_uniform('C', 0, 4)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            solver = trial.suggest_categorical('solver', ['liblinear'])
            model = LogisticRegression(C=C, solver=solver,penalty=penalty)
        elif model_name == 'random_forest':
            n_estimators = trial.suggest_int('n_estimators', 30, 500)
            max_depth = trial.suggest_int('max_depth', 1, 15)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        elif model_name == 'naive_bayes':
            model = GaussianNB()
        elif model_name == 'support_vector_machine':
            C = trial.suggest_uniform('C', 0.0, 10)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            gamma = trial.suggest_uniform('gamma', 0.0, 10)
            model = SVC(C=C, kernel=kernel, gamma=gamma)
        elif model_name == 'gradient_boosting':
            n_estimators = trial.suggest_int('n_estimators', 30, 500)
            max_depth = trial.suggest_int('max_depth', 1, 15)
            max_features = trial.suggest_uniform('max_features', 1, np.sqrt(train_data.shape[1]))
            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               max_features=max_features)
        elif model_name == 'decision_tree':
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            max_depth = trial.suggest_int('max_depth', 1, 15)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        elif model_name == 'knn':
            n_neighbors = trial.suggest_int('n_neighbors', 1, 20)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Evaluate the model
        model.fit(train_data, train_labels)
        tst_pred = model.predict(test_data)
        mcc = matthews_corrcoef(test_labels, tst_pred)
        return mcc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params

    if model_name == 'logistic_regression':
        best_model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'],solver=best_params['solver'],)
    elif model_name == 'random_forest':
        best_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                            max_depth=best_params['max_depth'])
    elif model_name == 'naive_bayes':
        best_model = GaussianNB()
    elif model_name == 'support_vector_machine':
        best_model = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
    elif model_name == 'gradient_boosting':
        best_model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                                max_depth=best_params['max_depth'],
                                                max_features=best_params['max_features'])
    elif model_name == 'decision_tree':
        best_model = DecisionTreeClassifier(criterion=best_params['criterion'], max_depth=best_params['max_depth'],
                                            min_samples_leaf=best_params['min_samples_leaf'])
    elif model_name == 'knn':
        best_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])

    best_model.fit(train_data, train_labels)

    return best_model
