import os
import warnings
import numpy as np
import pandas as pd
from pyexplainer.pyexplainer_pyexplainer import PyExplainer
from utilities.timewiseCV import time_wise_CV
from utilities.AutoSpearman import AutoSpearman
from sklearn.ensemble import RandomForestClassifier
from  utilities.CFS import cfs
import dalex as dx


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
    data_time=data['commitTime']
    #remove label
    all_cols = data.columns
    for col in all_cols:
        if col in ['bug', 'commitTime', 'commitdate']:
            data = data.drop(col, axis=1)
    # Remove Correlation and Redundancy(AotuSpearman)
    data_feature = AutoSpearman(data, correlation_threshold=0.7, correlation_method='spearman', VIF_threshold=5)
    ## Remove feature interaction(CFS)
    #idx = cfs(data_feature, data_label)
    data = pd.concat((data_time,data_feature, data_effort, data_label), axis=1)
    # log transformation
    cols_to_normalize = data.columns.difference(['commitTime','fix','effort','bug'])
    data[cols_to_normalize] = np.log(data[cols_to_normalize] + 1)
    return data

def time_wise_fold_divided(fold,train_folds,test_folds):
    #train
    train_label = train_folds[fold]['bug']
    train_data = train_folds[fold].drop(['effort','bug'], axis=1)
    #test
    LOC = test_folds[fold]['effort']
    test_label = test_folds[fold]['bug']
    test_data = test_folds[fold].drop(['effort','bug'], axis=1)
    indep = test_data.columns
    dep = 'bug'
    return train_data,train_label,test_data,test_label,LOC,indep,dep



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    save_path = r'./result-importance/'

    project_names = sorted(os.listdir('./dataset/'))
    path = os.path.abspath('./dataset/')
    pro_num = len(project_names)

    column_name = ['commitTime', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp',
                   'rexp', 'sexp', 'bug']
    for i in range(0, pro_num):
        # read data
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        project_name = project_name[:-4]
        data = data[column_name]
        #data preprocessing
        data= preprocessing(data)
        # save scores
        scores = {'original': [], 'rus': [], 'rom': [], 'smo': []}
        #time wise
        gap = 2
        train_folds, test_folds, _ = time_wise_CV(data, gap)
        for fold in range(len(train_folds)):
            train_data, train_label, test_data, test_label, LOC,indep, dep = time_wise_fold_divided(fold,train_folds,test_folds)
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
                #PyExplainer explainer
                model = RandomForestClassifier(n_estimators=200, random_state=1)
                model.fit(n_X, n_y)
                class_label = [0, 1]
                py_explainer = PyExplainer(n_X, n_y, indep, dep, model, class_label)
                #PyExplainer explain: one instance
                sample_explain_index = 7
                X_explain = test_data.iloc[[sample_explain_index]]
                y_explain = test_label.iloc[[sample_explain_index]]
                rules=py_explainer.explain(X_explain, y_explain, search_function='crossoverinterpolation', top_k=3, max_rules=30, max_iter=5, cv=5)
                #draw figure
                py_explainer.visualise(rules)
                #save figure
        print(f"{project_name} is okay~")
    print('done!')