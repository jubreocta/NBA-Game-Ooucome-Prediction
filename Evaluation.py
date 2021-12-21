import numpy as np
import time
import itertools
import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import random as r
r.seed(42)
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA

def greater_than_less_than(home_array, away_array):
    '''returns 1 if the home teams value is greater, 0 if it is less and
-1 if the values are too close to differentiate between the values'''
    return np.where(home_array - away_array > 0.00001, 1,
                    np.where(home_array - away_array < -0.00001, 0, -1))

def actual_result(dataset):
    '''returns the outcome of of each game in a dataset'''
    actual = greater_than_less_than(dataset['Home Score'], dataset['Away Score'])
    return actual

def equal_mean(actual, array2):
    '''calculates the mean accuracy in comparison to actual match values'''
    equal = 0
    total = 0
    for value_index in range(len(actual)):
        total+=1
        if actual[value_index] == array2[value_index]:
            equal+=1
    return equal / total

def equal_mean_known(array1, array2):
    '''calculates the mean accuracy in comparison to actual match values
    with only values that are actually ranks'''
    equal = 0
    total = 0
    for value_index in range(len(array1)):
        if array1[value_index] == -1 or array2[value_index] == -1:
            True
        else:
            total+=1
            if array1[value_index] == array2[value_index]:
                equal+=1
    return equal / total

def do_rankings_vote(dataset,type_ = 'all'):
    '''does ranking for all seasonal ranking algorithms and returns the accuracy'''
    actual = actual_result(dataset)
    if type_ == 'all':
        win_percentage = greater_than_less_than(dataset['home_win_percentage'],
                                        dataset['away_win_percentage'])
        colley = greater_than_less_than(dataset['colley_home_rating'],
                                        dataset['colley_away_rating'])
        massey_overall = greater_than_less_than(dataset['massey_home_overall_rating'],
                                        dataset['massey_away_overall_rating'])
        massey_off = greater_than_less_than(dataset['massey_home_offensive_rating'],
                                        dataset['massey_away_offensive_rating'])
        massey_dff = greater_than_less_than(dataset['massey_home_defensive_rating'],
                                        dataset['massey_away_defensive_rating'])
        od_overall = greater_than_less_than(dataset['od_home_overall_rating'],
                                        dataset['od_away_overall_rating'])
        od_off = greater_than_less_than(dataset['od_home_offensive_rating'],
                                        dataset['od_away_offensive_rating'])
        od_dff = greater_than_less_than(dataset['od_home_defensive_rating'],
                                        dataset['od_away_defensive_rating'])
        markov1 = greater_than_less_than(dataset['markov_home_rating1'],
                                        dataset['markov_away_rating1'])
        markov2 = greater_than_less_than(dataset['markov_home_rating2'],
                                        dataset['markov_away_rating2'])
        markov3 = greater_than_less_than(dataset['markov_home_rating3'],
                                        dataset['markov_away_rating3'])
        
        results = []
        results.append(len(dataset))
        results.append(equal_mean(actual, win_percentage))
        results.append(equal_mean(actual, colley))
        results.append(equal_mean(actual, massey_overall))
        results.append(equal_mean(actual, massey_off))
        results.append(equal_mean(actual, massey_dff))
        results.append(equal_mean(actual, od_overall))
        results.append(equal_mean(actual, od_off))
        results.append(equal_mean(actual, od_dff))
        results.append(equal_mean(actual, markov1))
        results.append(equal_mean(actual, markov2))
        results.append(equal_mean(actual, markov3))
    return results




def do_timed_vote(dataset,type_ = 'all'):
    '''does ranking for all timed ranking algorithms and returns the accuracy'''
    actual = actual_result(dataset)
    results = []
    if type_ == 'all':
        for i in range(1,8):
            colley = greater_than_less_than(dataset['colley_home_' + str(i) + '_month'],                   
                                            dataset['colley_away_' + str(i) + '_month'])
            massey_overall = greater_than_less_than(dataset['massey_home_overall_' + str(i) + '_month'],
                                            dataset['massey_away_overall_' + str(i) + '_month'])
            massey_off = greater_than_less_than(dataset['massey_home_offensive_' + str(i) + '_month'],
                                            dataset['massey_away_offensive_' + str(i) + '_month'])
            massey_dff = greater_than_less_than(dataset['massey_home_defensive_' + str(i) + '_month'],
                                            dataset['massey_away_defensive_' + str(i) + '_month'])
            od_overall = greater_than_less_than(dataset['od_home_overall_' + str(i) + '_month'],
                                            dataset['od_away_overall_' + str(i) + '_month'])
            od_off = greater_than_less_than(dataset['od_home_offensive_' + str(i) + '_month'],
                                            dataset['od_away_offensive_' + str(i) + '_month'])
            od_dff = greater_than_less_than(dataset['od_home_defensive_' + str(i) + '_month'],
                                            dataset['od_away_defensive_' + str(i) + '_month'])
            markov1 = greater_than_less_than(dataset['markov1_home_' + str(i) + '_month'],
                                            dataset['markov1_away_' + str(i) + '_month'])
            markov2 = greater_than_less_than(dataset['markov2_home_' + str(i) + '_month'],
                                            dataset['markov2_away_' + str(i) + '_month'])
            markov3 = greater_than_less_than(dataset['markov3_home_' + str(i) + '_month'],
                                            dataset['markov3_away_' + str(i) + '_month'])
            
            results.append(equal_mean(actual, colley))
            results.append(equal_mean(actual, massey_overall))
            results.append(equal_mean(actual, massey_off))
            results.append(equal_mean(actual, massey_dff))
            results.append(equal_mean(actual, od_overall))
            results.append(equal_mean(actual, od_off))
            results.append(equal_mean(actual, od_dff))
            results.append(equal_mean(actual, markov1))
            results.append(equal_mean(actual, markov2))
            results.append(equal_mean(actual, markov3))
    return results



def brute_force_lr(predictors_dataset, Y):
    '''implements the exhaustive search for the best feature combination.'''
    best=[('a',0,0,0)]*20
    estimators_LR = []
    estimators_LR.append(('rescale', MinMaxScaler()))
    estimators_LR.append(('LR_sag', LogisticRegression(max_iter = 10000)))
    pipe = Pipeline(estimators_LR)
    kfold = KFold(n_splits=10)
    scoring = 'accuracy'
    start_column = 'a'
    '''
    for index in range(43, len(predictors_dataset.columns)):
        print(index)
        if index<6 or index>39:
            for combination in itertools.combinations(predictors_dataset.columns, index):
                columns = [i for i in combination]
                data = predictors_dataset[columns]
                X = data.values
                X = np.nan_to_num(X)
                results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
                if results.mean() > best[-1][2]:
                    best[-1] = (combination, index, results.mean(), results.std())
                    best = sorted(best, key = lambda x:x[2], reverse = True)
                    with open('bruteforce.txt', 'w') as top20:
                        for value in best:
                            top20.write(str(value)+'\n')
        else:
            for number in range(20000):
                if number%100 == 0:
                    print(number)
                combination = r.sample(predictors_dataset.columns.tolist(), index)
                data = predictors_dataset[combination]
                X = data.values
                X = np.nan_to_num(X)
                results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
                if results.mean() > best[-1][2]:
                    best[-1] = (combination, index, results.mean(), results.std())
                    best = sorted(best, key = lambda x:x[2], reverse = True)
                    with open('bruteforce.txt', 'w') as top20:
                        for value in best:
                            top20.write(str(value)+'\n')
    '''
    #pulling from a text file because this process takes long and
    #there is no time to wait to the end
    #we simply grab the best so far to move on
    best_to_return = []
    column_names = predictors_dataset.columns.to_list()
    with open('bruteforce.txt', 'r') as top20:
        for line in top20:
            new_features = line.strip().split("'],")[0].split("', '")
            new_features = [column_names.index(i) for i in new_features]
            new_features = sorted(new_features)
            index = line.strip().split("'],")[1].strip().split(", ")
            best_to_return.append((new_features, index[0], index[1], index[2]))
    return best_to_return
                            
def SelectKBest_(predictors_dataset, Y):
    '''uses the sklearn selectkbest to iteratively select the best
    feature combination for each number of features '''
    start_time = time.time()
    highest_accuracy = 0
    best=[]
    for index in range(1, len(predictors_dataset.columns)+1):
        print(index)
        estimators_LR = []
        estimators_LR.append(('rescale', MinMaxScaler()))
        estimators_LR.append(('select_best_chi', SelectKBest(score_func=chi2, k=index)))
        estimators_LR.append(('LR_sag', LogisticRegression(max_iter = 10000)))
        pipe = Pipeline(estimators_LR)
        kfold = KFold(n_splits=10, shuffle = True, random_state=222)
        scoring = 'accuracy'
        X = predictors_dataset.values
        X = np.nan_to_num(X)
        pipe.fit(X, Y)
        results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
        #get only improvements on previous results for time monitoring
        if results.mean() > highest_accuracy:
            time_diff = time.time() - start_time
            highest_accuracy = results.mean()
            mask = pipe['select_best_chi'].get_support() #list of booleans
            new_features = [] # The list of your K best features
            for bool_, feature in zip(mask, range(len(predictors_dataset.columns))):
                if bool_:
                    new_features.append(feature)
            best.append((new_features, index, results.mean(), results.std(), time_diff))
    return best


def SequentialForwardSelection(predictors_dataset, Y):
    '''implements the greedy search algorithm sequential forward selection'''
    best = []
    initial_features = predictors_dataset.columns.tolist()
    best_features = []
    count = 0
    for count in range(len(initial_features)):
        print(count+1)
        remaining_features = list(set(initial_features)-set(best_features))
        mean = 0
        for new_column in remaining_features:
            feat = copy.deepcopy(best_features)
            feat.append(new_column)
            estimators_LR = []
            estimators_LR.append(('rescale', MinMaxScaler()))
            estimators_LR.append(('LR_sag', LogisticRegression(max_iter = 10000)))
            pipe = Pipeline(estimators_LR)
            kfold = KFold(n_splits=10)
            scoring = 'accuracy'
            X = predictors_dataset[feat].values
            X = np.nan_to_num(X)
            pipe.fit(X, Y)
            results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
            check.append((feat, results.mean()))
            if results.mean() > mean:
                mean = results.mean()
                stdev = results.std()
                new_best = new_column
        best_features.append(new_best)
        column_index = sorted([initial_features.index(i) for i in best_features])
        best.append((column_index, count+1, mean, stdev))
    best = sorted(best, key = lambda x: x[2], reverse = True)
    return best



def SequentialBackwardSelection(predictors_dataset, Y):
    '''implements the greedy search algorithm sequential backward selection'''
    best = []
    initial_features = predictors_dataset.columns.tolist()
    best_features = predictors_dataset.columns.tolist()
    count = 0
    for count in range(len(initial_features)-1,0,-1):
        print(count+1)
        remaining_features = copy.deepcopy(best_features)
        mean = 0
        for new_column in remaining_features:
            feat = copy.deepcopy(best_features)
            feat.remove(new_column)
            estimators_LR = []
            estimators_LR.append(('rescale', MinMaxScaler()))
            estimators_LR.append(('LR_sag', LogisticRegression(max_iter = 10000)))
            pipe = Pipeline(estimators_LR)
            kfold = KFold(n_splits=10)
            scoring = 'accuracy'
            X = predictors_dataset[feat].values
            X = np.nan_to_num(X)
            pipe.fit(X, Y)
            results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
            if results.mean() > mean:
                mean = results.mean()
                stdev = results.std()
                new_best = feat
        best_features = new_best
        column_index = sorted([initial_features.index(i) for i in best_features])
        best.append((column_index, count, mean, stdev))
    best = sorted(best, key = lambda x: x[2], reverse = True)
    return best

def PCA_(predictors_dataset, Y):
    '''uses the sklearn PCA to iteratively select the best
    feature combination for each number of principal components '''
    start_time = time.time()
    highest_accuracy = 0
    best=[]
    for index in range(1, len(predictors_dataset.columns)+1):
        print(index)
        estimators_LR = []
        estimators_LR.append(('rescale', MinMaxScaler()))
        estimators_LR.append(('PCA',  PCA(n_components=index)))
        estimators_LR.append(('LR_sag', LogisticRegression(max_iter = 10000)))
        pipe = Pipeline(estimators_LR)
        kfold = KFold(n_splits=10)
        scoring = 'accuracy'
        X = predictors_dataset.values
        X = np.nan_to_num(X)
        pipe.fit(X,Y)
        results = cross_val_score(pipe, X, Y, cv=kfold, scoring=scoring)
        if results.mean() > highest_accuracy:
            time_diff = time.time() - start_time
            highest_accuracy = results.mean()
            best.append((index,results.mean(), results.std(), time_diff))
    #print(pipe['PCA'].explained_variance_ratio_)
    cumsum = np.cumsum(pipe['PCA'].explained_variance_ratio_)
    #print(cumsum)
    plt.xticks(ticks = range(0,46,4), labels = range(1,47,4))
    plt.xlabel('number of components', size = 10)
    plt.ylabel('cummulative explained variance', size = 10)
    plt.title('Cummulative Sum of Variance Explained by Principal Components')
    plt.plot(np.cumsum(pipe['PCA'].explained_variance_ratio_))
    for i, v in enumerate(cumsum):
        if i in [0, 1, 2, 4]:
            plt.text(i+1.75, v, "%.2f" % v, ha="center", size = 8)        
        elif i in [8, 12, 16, 30]:
            plt.text(i, v-0.025, "%.2f" % v, ha="center", size = 8)
    plt.show()
    return best

def MasterModel(dataset, how):
    '''Does the final validation on the dataset by training on 3
    previous seasons to test on the current season.'''
    final_result = dict() #stores all data to be returned
    #feature selecting list
    feature_selection = []
    feature_selection.append(('bs only',
                              FunctionTransformer(backward_selection_only)))
    feature_selection.append(('bs / acute',
                              FunctionTransformer(backward_selection_with_acute)))
    feature_selection.append(('bs / chronic',
                              FunctionTransformer(backward_selection_with_chronic)))
    feature_selection.append(('bs / acwr',
                              FunctionTransformer(backward_selection_with_ratio)))
    feature_selection.append(('fs only',
                              FunctionTransformer(forward_selection_only)))    
    feature_selection.append(('fs / acute',
                              FunctionTransformer(forward_selection_with_acute)))
    feature_selection.append(('fs / chronic',
                              FunctionTransformer(forward_selection_with_chronic)))
    feature_selection.append(('fs / acwr',
                              FunctionTransformer(forward_selection_with_ratio)))
    feature_selection.append(('bf only',
                              FunctionTransformer(brute_force_only)))
    feature_selection.append(('bf / acute',
                              FunctionTransformer(brute_force_with_acute)))
    feature_selection.append(('bf / chronic',
                              FunctionTransformer(brute_force_with_chronic)))
    feature_selection.append(('bf / acwr',
                              FunctionTransformer(brute_force_with_ratio)))
    models = []
    models.append(('liblinear', LogisticRegression(max_iter = 10000,
                                                solver = 'liblinear')))
    models.append(('newton_cg', LogisticRegression(max_iter = 10000,
                                                solver = 'newton-cg')))
    models.append(('lbfgs', LogisticRegression(max_iter = 10000,
                                                solver = 'lbfgs')))
    models.append(('saga', LogisticRegression(max_iter = 10000,
                                                solver = 'saga')))
    if how == '3PreviousSeasons':
        #stores all data to be returned
        for technique in feature_selection:
            for model in models:
                results=[]
                no_of_columns = len(dataset.columns)
                seasons = dataset.season.unique()
                for i in range(3, len(seasons)):
                    #train with previous 3 seasons. loop is for i in
                    #range(3, len(seasons)) accuracy of 67.15 with backward
                    #selection max is 70.6
                    train = dataset[dataset.season.isin([seasons[j] for j in range(i-3, i)])]
                    
                    test = dataset[dataset.season == seasons[i]]
                    #print(test.season.unique(), train.season.unique())
                    X_test  = test.iloc[:, :no_of_columns-2].values
                    Y_test  = test.Y.values
                    X_train = train.iloc[:, :no_of_columns-2].values
                    Y_train = train.Y.values
                    #print(X_test.shape, Y_test.shape, X_train.shape, Y_train.shape)
                    estimators_LR = []
                    estimators_LR.append(('nan_to_num',
                                          SimpleImputer(strategy='constant',
                                                        fill_value = 0)))
                    estimators_LR.append(('rescale', MinMaxScaler()))
                    estimators_LR.append(technique)
                    estimators_LR.append(model)
                    pipe = Pipeline(estimators_LR)
                    #fit
                    pipe.fit(X_train, Y_train)
                    #modelling
                    result = pipe.score(X_test, Y_test)
                    results.append(result)
                final_result[technique[0]+' / '+model[0]] = results
    elif how == 'AllPreviousSeasons':
        for technique in feature_selection:
            for model in models:
                results=[]
                no_of_columns = len(dataset.columns)
                seasons = dataset.season.unique()
                for i in range(2, len(seasons)):
                    #train with all previous seasons skipping first season.
                    #loop is for i in range(2, len(seasons))
                    #accuracy of 67.27 with backward selection max is 70.6
                    train = dataset[dataset.season.isin([seasons[j] for j in range(1, i)])]

                    test = dataset[dataset.season == seasons[i]]
                    #print(test.season.unique(), train.season.unique())
                    X_test  = test.iloc[:, :no_of_columns-2].values
                    Y_test  = test.Y.values
                    X_train = train.iloc[:, :no_of_columns-2].values
                    Y_train = train.Y.values
                    #print(X_test.shape, Y_test.shape, X_train.shape, Y_train.shape)
                    estimators_LR = []
                    estimators_LR.append(('nan_to_num',
                                          SimpleImputer(strategy='constant',
                                                        fill_value = 0)))
                    estimators_LR.append(('rescale', MinMaxScaler()))
                    estimators_LR.append(technique)
                    estimators_LR.append(model)
                    pipe = Pipeline(estimators_LR)
                    #fit
                    pipe.fit(X_train, Y_train)
                    #modelling
                    result = pipe.score(X_test, Y_test)
                    results.append(result)
                final_result[technique[0]+' / '+model[0]] = results
    results = np.array(results)
    #print(results.mean(), results.std())
    return final_result

def backward_selection_columns():
    '''stores the index of the best backward selection features'''
    return [0, 1, 2, 3, 4, 7, 9, 10, 13,
            16, 17, 20, 21, 23, 25, 26,
            28, 29, 30, 31, 32, 35,
            39, 40, 41, 42, 43
            ]
def forward_selection_columns():
    '''stores the index of the best forward selection features'''
    return[0, 2, 4, 8, 9, 11, 12, 14,
           15, 16, 17, 19, 21, 22, 25,
           26, 30, 33, 34, 36, 38, 41
           ]
def brute_force_columns():
    '''stores the index of the best brute force selected features'''
    return [2, 4, 5, 6, 9, 10, 12, 13,
            14, 17, 22, 23, 24, 25, 30,
            31, 34, 36, 38, 39
            ]
def acute_columns():
    '''acute load features'''
    return [44, 45]
def chronic_columns():
    '''chronic load features'''
    return [46, 47]
def ratio_columns():
    '''acwr features'''
    return [48, 49]

def backward_selection_only(X):
    '''transforms dataset to backward selection features only'''
    transformation = backward_selection_columns()
    return X[:, transformation]
def backward_selection_with_acute(X):
    '''transforms dataset to backward selection and acute load features only'''
    transformation = backward_selection_columns()
    transformation.extend(acute_columns())
    return X[:, transformation]
def backward_selection_with_chronic(X):
    '''transforms dataset to backward selection and chronic load features only'''
    transformation = backward_selection_columns()
    transformation.extend(chronic_columns())
    return X[:, transformation]
def backward_selection_with_ratio(X):
    '''transforms dataset to backward selection and acwr features only'''
    transformation = backward_selection_columns()
    transformation.extend(ratio_columns())
    return X[:, transformation]

def forward_selection_only(X):
    '''transforms dataset to forward selection features only'''
    transformation = forward_selection_columns()
    return X[:, transformation]
def forward_selection_with_acute(X):
    '''transforms dataset to forward selection and acute load features only'''
    transformation = forward_selection_columns()
    transformation.extend(acute_columns())
    return X[:, transformation]
def forward_selection_with_chronic(X):
    '''transforms dataset to forward selection and chronic load features only'''
    transformation = forward_selection_columns()
    transformation.extend(chronic_columns())
    return X[:, transformation]
def forward_selection_with_ratio(X):
    '''transforms dataset to forward selection and acwr features only'''
    transformation = forward_selection_columns()
    transformation.extend(ratio_columns())
    return X[:, transformation]

def brute_force_only(X):
    '''transforms dataset to brute force selected features only'''
    transformation = brute_force_columns()
    return X[:, transformation]
def brute_force_with_acute(X):
    '''transforms dataset to brute force selected and acute load features only'''
    transformation = brute_force_columns()
    transformation.extend(acute_columns())
    return X[:, transformation]
def brute_force_with_chronic(X):
    '''transforms dataset to brute force selected and chronic load features only'''
    transformation = brute_force_columns()
    transformation.extend(chronic_columns())
    return X[:, transformation]
def brute_force_with_ratio(X):
    '''transforms dataset to brute force selected and acwr features only'''
    transformation = brute_force_columns()
    transformation.extend(ratio_columns())
    return X[:, transformation]
