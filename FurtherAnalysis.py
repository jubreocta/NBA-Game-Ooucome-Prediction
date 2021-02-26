from Evaluation import backward_selection_only, brute_force_with_ratio
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import pandas as pd
import numpy as np
import random as r
r.seed(42)

def probabilities(train, test, type_):
    '''returns the prediction probabilities from predictions training
    using only the previous 3 seasons
    '''
    no_of_columns = len(train.columns)
    X_test  = test.iloc[:, :no_of_columns-2].values
    Y_test  = test.Y.values
    X_train = train.iloc[:, :no_of_columns-2].values
    Y_train = train.Y.values
    #print(X_test.shape, Y_test.shape, X_train.shape, Y_train.shape)
    estimators_LR = []
    estimators_LR.append(('nan_to_num',
                          SimpleImputer(strategy='constant',
                                        fill_value = 0)
                          )
                         )
    estimators_LR.append(('rescale', MinMaxScaler()
                          )
                         )
    if type_ == '3PreviousSeasons':
        estimators_LR.append(('bs only',
                                  FunctionTransformer(brute_force_with_ratio)))
        estimators_LR.append(('liblinear', LogisticRegression(max_iter = 10000,
                                                    solver = 'liblinear')))
    elif type_ == 'AllPreviousSeasons':
        estimators_LR.append(('bs only',
                                  FunctionTransformer(backward_selection_only)))
        estimators_LR.append(('newtoncg', LogisticRegression(max_iter = 10000,
                                                    solver = 'newton-cg')))
        
    pipe = Pipeline(estimators_LR)
    #fit
    pipe.fit(X_train, Y_train)
    return pipe.predict_proba(X_test)[:, 1]


def score_differential():
    '''returns the score differential for all games in the 2019-2020 season'''
    last_season = pd.read_csv(r'Data/1 Season/2019-2020.csv')
    diff =  np.array([int(i.split(' - ')[0]) - int(i.split(' - ')[1]) \
            for i in last_season.Result]).reshape(-1, 1)
    last_season['Scaled Point Differential'] = MinMaxScaler().fit_transform(diff)
    return last_season


def availability_analysis(dataset, top, type_):
    '''performs availability analysis by using the other functions in
    this script to return a dataset with a suprise column sorted to show
    the top (function argument) suprise games'''
    last_season = score_differential()
    test = dataset[dataset.season == '2019-2020']
    if type_ == '3PreviousSeasons':
        train = dataset[dataset.season.isin(['2016-2017', '2017-2018',
                                             '2018-2019',
                                             ])]
        last_season['Prediction Probabilities'] = probabilities(train,
                                                                test, type_)
    elif type_ == 'AllPreviousSeasons':
        train = dataset[dataset.season.isin([  '2005-2006', '2006-2007',
                                               '2007-2008', '2008-2009',
                                               '2009-2010', '2010-2011',
                                               '2011-2012', '2012-2013',
                                               '2013-2014', '2014-2015',
                                               '2015-2016', '2016-2017',
                                               '2017-2018', '2018-2019'
                                               ])]
        last_season['Prediction Probabilities'] = probabilities(train,
                                                                test, type_)
    last_season['Surprise'] = abs(last_season['Scaled Point Differential'] - \
                                 last_season['Prediction Probabilities'])
    last_season = last_season.sort_values('Surprise', ascending = False)
    last_season = last_season.drop('Season', axis = 1)
    last_season = last_season.drop('Overtime', axis = 1)
    print(last_season.head(top))
    topN = last_season.iloc[0:top,:]
    topN.to_csv('Results/TopSurprise'+ type_+'.csv', index = False)

def data_to_plot(dataset, type_):
    '''returns a rolling average of prediction accuracy over each season
    '''
    final_dict = dict()#dictionary converted to dataframe
    estimators_LR = []
    no_of_columns = len(dataset.columns)
    seasons = dataset.season.unique()
    if type_ == '3PreviousSeasons':
        for i in range(3, len(seasons)):
            train = dataset[dataset.season.isin([seasons[j] for j in range(i-3, i)])]      
            test = dataset[dataset.season == seasons[i]]
            X_test  = test.iloc[:, :no_of_columns-2].values
            Y_test  = test.Y.values
            X_train = train.iloc[:, :no_of_columns-2].values
            Y_train = train.Y.values
            estimators_LR = []
            estimators_LR.append(('nan_to_num',
                                    SimpleImputer(strategy='constant',
                                                fill_value = 0)
                                  )
                                 )
            estimators_LR.append(('rescale', MinMaxScaler()))
            estimators_LR.append(('bf / acwr',
                                  FunctionTransformer(brute_force_with_ratio)))
            estimators_LR.append(('liblinear',
                                  LogisticRegression(max_iter = 10000,
                                                    solver = 'liblinear')
                                  )
                                 )
            pipe = Pipeline(estimators_LR)
            pipe.fit(X_train, Y_train)
            predictions = pipe.predict(X_test)
            total = 0
            correct_pred = 0
            rolling_average = []
            for index in range(len(predictions)):
                if predictions[index] == Y_test[index]:
                    correct_pred+=1
                total+=1
                rolling_average.append(correct_pred / total)
            final_dict[seasons[i]] = rolling_average

    elif type_ == 'AllPreviousSeasons':
        for i in range(2, len(seasons)):
            train = dataset[dataset.season.isin([seasons[j] for j in range(1, i)])]    
            test = dataset[dataset.season == seasons[i]]
            X_test  = test.iloc[:, :no_of_columns-2].values
            Y_test  = test.Y.values
            X_train = train.iloc[:, :no_of_columns-2].values
            Y_train = train.Y.values
            estimators_LR = []
            estimators_LR.append(('nan_to_num',
                                    SimpleImputer(strategy='constant',
                                                fill_value = 0)
                                  )
                                 )
            estimators_LR.append(('rescale', MinMaxScaler()))
            estimators_LR.append(('bs only',
                                  FunctionTransformer(backward_selection_only)))
            estimators_LR.append(('liblinear',
                                  LogisticRegression(max_iter = 10000,
                                                    solver = 'newton-cg')
                                  )
                                 )
            pipe = Pipeline(estimators_LR)
            pipe.fit(X_train, Y_train)
            predictions = pipe.predict(X_test)
            total = 0
            correct_pred = 0
            rolling_average = []
            for index in range(len(predictions)):
                if predictions[index] == Y_test[index]:
                    correct_pred+=1
                total+=1
                rolling_average.append(correct_pred / total)
            final_dict[seasons[i]] = rolling_average
    rolling_average = pd.DataFrame(dict([ (k,pd.Series(v)) \
                                          for k,v in final_dict.items() ]))
    return rolling_average
