import numpy as np
import pandas as pd
from eli5.sklearn import PermutationImportance
from matplotlib import pyplot as plt
import shap
import os
import warnings
from utilities.timewiseCV import time_wise_CV
from utilities.AutoSpearman import AutoSpearman
from sklearn.ensemble import RandomForestClassifier
import dalex as dx
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utilities.tuneParameters import tune

sampling_methods = {
    "none": None,
    'rus': RandomUnderSampler(random_state=0),
    'rom': RandomOverSampler(random_state=0),
    'smo': SMOTE(random_state=0),
}

def preprocessing(data):
    #preserve label and effort
    data_label = data['bug'].to_frame()
    data_effort = (data['la'] + data['ld']).to_frame()
    data_effort.columns = ['effort']
    data_time = data['commitTime']
    #remove label
    all_cols = data.columns
    for col in all_cols:
        if col in ['bug', 'commitTime', 'commitdate']:
            data = data.drop(col, axis=1)
    # Remove Correlation and Redundancy(AotuSpearman)
    data_feature = AutoSpearman(data, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5)
    ## Remove feature interaction(CFS)
    #idx = cfs(data_feature, data_label)
    num_feature=data_feature.shape[1]
    data = pd.concat((data_time, data_feature, data_effort, data_label), axis=1)
    # log transformation
    cols_to_normalize = data.columns.difference(['commitTime','fix','effort','bug'])
    data[cols_to_normalize] = np.log(data[cols_to_normalize] + 1)
    return data, num_feature

def time_wise_fold_divided(fold,train_folds,test_folds,gap_folds):
    #train
    train_label = train_folds[fold]['bug']
    train_data = train_folds[fold].drop(['effort','bug'], axis=1)
    #test
    LOC = test_folds[fold]['effort']
    test_label = test_folds[fold]['bug']
    test_data = test_folds[fold].drop(['effort','bug'], axis=1)
    #gap
    gap_label = gap_folds[fold]['bug']
    gap_data = gap_folds[fold].drop(['effort', 'bug'], axis=1)
    return train_data,train_label,test_data,test_label,gap_data,gap_label,LOC

def save_shap_results_to_csv(results, project_name,feature_names,model_name):
    for key, value in results.items():
        df = pd.DataFrame(value)
        # Drop rows with NaN values
        df = df.dropna()
        # Reset the index
        df = df.reset_index(drop=True)
        df.columns = feature_names
        outpath = f'./output/global/shap_scores/{project_name}-{model_name}-{key}.csv'
        df.to_csv(outpath, index=True, header=True)

def save_permutation_results_to_csv(results, project_name,feature_names,model_name):
    for key, value in results.items():
        df = pd.DataFrame(value)
        # Drop rows with NaN values
        df = df.dropna()
        # Reset the index
        df = df.reset_index(drop=True)
        df.columns = feature_names
        outpath = f'./output/global/permutation_scores/{project_name}-{model_name}-{key}.csv'
        df.to_csv(outpath, index=True, header=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    save_path = r'./result-importance/'
    project_names = sorted(os.listdir('./dataset/'))
    path = os.path.abspath('./dataset/')
    pro_num = len(project_names)
    column_name = ['commitTime', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp',
                   'rexp', 'sexp', 'bug']
    model_names =['naive_bayes', 'support_vector_machine', 'gradient_boosting', 'decision_tree', 'knn', 'random_forest', 'logistic_regression']
    for model_name in model_names:
        for i in range(0, pro_num):
            # read data
            project_name = project_names[i]
            file = os.path.join(path, project_name)
            data = pd.read_csv(file)
            project_name = project_name[:-4]
            data = data[column_name]
            #data preprocessing
            data,num_feature= preprocessing(data)
            # save feature importance
            results_shap = {key: np.zeros(shape=(0, num_feature)) for key in sampling_methods.keys()}
            results_permutation = {key: np.zeros(shape=(0, num_feature)) for key in sampling_methods.keys()}
            #time wise
            gap = 2
            train_folds, test_folds, gap_folds = time_wise_CV(data, gap)
            for fold in range(len(train_folds)):
                train_data, train_label, test_data, test_label, gap_data, gap_label, LOC = time_wise_fold_divided(fold,train_folds,test_folds,gap_folds)
                # ensure train data: the number of defect > non defect, only one class
                if len(np.unique(train_label)) < 2 or list(train_label).count(1) < 6 or list(train_label).count(1) > list(train_label).count(0):
                    continue
                for method, sampler in sampling_methods.items():
                    if sampler is None:
                        n_X, n_y = train_data, train_label
                    else:
                        n_X, n_y = sampler.fit_resample(train_data, train_label)
                    # ensure test data is not single class
                    if list(n_y).count(1) < 2 or list(n_y).count(0) < 2:
                        break
                    #train model(tune para)
                    model = tune(n_X, n_y,gap_data, gap_label,model_name)

                    # calculate feature importance
                    perm = PermutationImportance(model).fit(test_data, test_label)
                    result_permutation = perm.feature_importances_
                    # merge fold feature importance
                    feature_names = test_data.columns
                    results_permutation[method] = np.vstack((results_permutation[method], result_permutation))
                    print(f"{fold}: {method} Permutation is okay~")

                    # shap_explain_global
                    if (model_name in ['naive_bayes','support_vector_machine','knn']):
                        explainer = shap.Explainer(model.predict, test_data)
                    elif (model_name in ['random_forest']):
                        shap_values = explainer(test_data).values
                    elif (model_name in ['gradient_boosting','decision_tree']):
                        explainer = shap.Explainer(model, check_additivity=False)
                    else:
                        explainer = shap.Explainer(model, test_data)

                    #calculate shap values
                    if( model_name in ['random_forest','decision_tree']):
                        shap_values = explainer.shap_values(test_data)[0]
                    elif(model_name in ['naive_bayes','support_vector_machine','knn']):
                        shap_values = explainer(test_data).values
                    else:
                        shap_values = explainer.shap_values(test_data)
                    #shap value predicted as defect proneness
                    feature_names = test_data.columns
                    shap_values_df = pd.DataFrame(shap_values, columns=feature_names)
                    # calculate the mean of shap_value
                    result_shap = np.abs(shap_values_df).mean()
                    results_shap[method] = np.vstack((results_shap[method], result_shap))
                    print(f"{fold}: Shap {method} is okay~")

            save_shap_results_to_csv(results_shap, project_name,feature_names,model_name)
            save_permutation_results_to_csv(results_permutation, project_name, feature_names, model_name)
            print(f"{project_name} is okay~")
        print(f"{model_name} running is okay~")