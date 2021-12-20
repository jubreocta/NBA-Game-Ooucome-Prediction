#standard Libraries
import copy
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import numpy as np
import datetime
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#created functions import
from Load import load_data
from Manipulations import home_away_different_teams, merge_data
from Fatigue import return_ewma_column, back2back
from RankingMethods import do_seasonal_ranking, do_timed_ranking
from Evaluation import do_rankings_vote, do_timed_vote, actual_result, brute_force_lr
from Evaluation import SelectKBest_, SequentialForwardSelection
from Evaluation import SequentialBackwardSelection, PCA_, MasterModel
from FurtherAnalysis import availability_analysis, data_to_plot

###################################################
###################################################
#0) Seasonal Ranking - Uncomment this to repopulate the 30TeamsSeasonalRankings folder
###################################################
###################################################
'''
print('#0) Seasonal Ranking - Commented out as better result was found')
pattern = re.compile(r'20[0-9][0-9]-20[0-9][0-9]')
DATA_PATH = 'Data/1 Season'
rank_votings = dict()
for file in os.listdir(DATA_PATH):
    if pattern.match(file):
        print(file)
        dataset = load_data(DATA_PATH + '/' + file)
        dataset = do_seasonal_ranking(dataset)
        season = file.split(".")[0]
        dataset.to_csv('.30TeamsSeasonalRankings/'+season+'.csv', index = False)
        rank_votings[file[:-4]] = do_rankings_vote(dataset)
ranks_voting = pd.DataFrame.from_dict(rank_votings)
ranks_voting['Type'] = ['DataPoints', 'WinPercentage',
                        'Colley',
                        'MasseyOverall', 'MasseyOffensive', 'MasseyDefensive',
                        'ODOverall', 'ODOffensive', 'ODDeffensive',
                        'Markov1', 'Markov2', 'Markov3']
ranks_voting.to_csv('Results/RankVotings.csv', index = False)
exit()
'''
###################################################
###################################################
#1) Avoiding cold start/ merging all datasets into 1 - Commented out as better result was found
###################################################
###################################################
'''
print('#1) Avoiding cold start/ merging all datasets into 1 - Commented out as better result was found')
DATA_PATH = 'Data/1 Season'
pattern = '20[0-9][0-9]-20[0-9][0-9]'
dataset = merge_data(DATA_PATH, pattern)
dataset = load_data(dataset, '')
dataset = do_seasonal_ranking(dataset)
seasons = dataset.Season.unique()
rank_votings = dict()
for season in seasons:
    data = dataset[dataset.Season == season]
    rank_votings[season] = do_rankings_vote(data)
ranks_voting = pd.DataFrame.from_dict(rank_votings)
ranks_voting['Type'] = ['DataPoints', 'WinPercentage',
                        'Colley',
                        'MasseyOverall', 'MasseyOffensive', 'MasseyDefensive',
                        'ODOverall', 'ODOffensive', 'ODDeffensive',
                        'Markov1', 'Markov2', 'Markov3']
ranks_voting.to_csv('Results/RankVotingsAvoidColdStart.csv', index = False)
'''
###################################################
###################################################
#2) Avoiding Cold Start/ Using only Previous season - Uncomment to repopulate PreviousSeasonRankings folder
###################################################
###################################################
'''
print('#2) Avoiding Cold Start/ Using only Previous season')
rank_votings = dict()
pattern = re.compile(r'20[0-9][0-9]-20[0-9][0-9]')
MASTER_DATA_PATH = 'Data/2 Seasons'
directory_list = os.listdir(MASTER_DATA_PATH)
for folder in directory_list:    
    for file in os.listdir(MASTER_DATA_PATH + '/' + folder):
        data = pd.read_csv(MASTER_DATA_PATH + '/' + folder + '/' + file)
        data['Season'] = file[:-4]
        data.to_csv(MASTER_DATA_PATH + '/' + folder + '/' + file, index = False)
 
    if pattern.match(folder):
        print(folder)
        DATA_PATH = MASTER_DATA_PATH + '/' + folder
        dataset = merge_data(DATA_PATH, pattern)
        dataset = load_data(dataset, '')
        dataset = do_seasonal_ranking(dataset)
        season = os.listdir(MASTER_DATA_PATH + '/' + folder)[-1][:-4]
        dataset = dataset[dataset.Season == season]
        dataset.to_csv('.PreviousSeasonRankings/'+season+'.csv', index = False)
        rank_votings[season] = do_rankings_vote(dataset)
ranks_voting = pd.DataFrame.from_dict(rank_votings)
ranks_voting['Type'] = ['DataPoints', 'WinPercentage',
                        'Colley',
                        'MasseyOverall', 'MasseyOffensive', 'MasseyDefensive',
                        'ODOverall', 'ODOffensive', 'ODDeffensive',
                        'Markov1', 'Markov2', 'Markov3']
ranks_voting.to_csv('Results/RankVotingsPreviousSeason.csv', index = False)
exit()
'''
###################################################
###################################################
#3) check for any advantage seperating home and away team -- no advantage
###################################################
###################################################
'''
print('#3) check for any advantage seperating home and away team')
rank_votings = dict()
pattern = re.compile(r'20[0-9][0-9]-20[0-9][0-9]')
MASTER_DATA_PATH = 'Data/2 Seasons'
directory_list = os.listdir(MASTER_DATA_PATH)
for folder in directory_list:
    if pattern.match(folder):
        print(folder)
        DATA_PATH = MASTER_DATA_PATH + '/' + folder
        dataset = merge_data(DATA_PATH, pattern)
        dataset = load_data(dataset, '')
        dataset = home_away_different_teams(dataset)
        dataset = do_seasonal_ranking(dataset)
        season = os.listdir(MASTER_DATA_PATH + '/' + folder)[-1][:-4]
        rank_votings[season] = do_rankings_vote(dataset)     
ranks_voting = pd.DataFrame.from_dict(rank_votings)
ranks_voting['Type'] = ['DataPoints', 'WinPercentage',
                        'Colley',
                        'MasseyOverall', 'MasseyOffensive', 'MasseyDefensive',
                        'ODOverall', 'ODOffensive', 'ODDeffensive',
                        'Markov1', 'Markov2', 'Markov3']
ranks_voting.to_csv('Results/RankVotings60Teams.csv', index = False)
'''
###################################################
###################################################
#4 csv of results from previous n months --no better results
###################################################
###################################################
'''
print('#4 csv of results from previous n months')
rank_votings = dict()
pattern = re.compile(r'20[0-9][0-9]-20[0-9][0-9]')
MASTER_DATA_PATH = 'Data/2 Seasons'
directory_list = os.listdir(MASTER_DATA_PATH)
for folder in directory_list:    
    print(folder)
    if pattern.match(folder):
        DATA_PATH = MASTER_DATA_PATH + '/' + folder
        dataset = merge_data(DATA_PATH, pattern)
        dataset = load_data(dataset, '')
        dataset = do_timed_ranking(dataset)
        season = os.listdir(MASTER_DATA_PATH + '/' + folder)[-1][:-4]
        rank_votings[season] = do_timed_vote(dataset)
ranks_voting = pd.DataFrame.from_dict(rank_votings)
Type = []
ranks = ['Colley',
         'MasseyOverall', 'MasseyOffensive', 'MasseyDefensive',
         'ODOverall', 'ODOffensive', 'ODDeffensive',
         'Markov1', 'Markov2', 'Markov3']
for i in range(1,8):
    for rank in ranks:
        Type.append(rank + '_' + str(i) + 'Months')
ranks_voting['Type'] = Type
ranks_voting.to_csv('Results/RankVotingsNMonths.csv', index = False)
'''
###################################################
###################################################
 #Feature Work
###################################################
###################################################
columns =  ['home_win_percentage', 'away_win_percentage',
            'colley_home_rating', 'colley_away_rating',
            'massey_home_overall_rating', 'massey_home_offensive_rating',
            'massey_home_defensive_rating', 'massey_away_overall_rating',
            'massey_away_offensive_rating', 'massey_away_defensive_rating',
            'od_home_overall_rating', 'od_home_offensive_rating',
            'od_home_defensive_rating', 'od_away_overall_rating',
            'od_away_offensive_rating', 'od_away_defensive_rating',
            'markov_home_rating1', 'markov_away_rating1',
            'markov_home_rating2', 'markov_away_rating2',
            'markov_home_rating3', 'markov_away_rating3']

DATA_PATH = '.30TeamsSeasonalRankings'
pattern = '20[0-9][0-9]-20[0-9][0-9]'
datasetS = merge_data(DATA_PATH, pattern)
Y = actual_result(datasetS)#ranking for a season
datasetS = datasetS[columns]
datasetS.columns = datasetS.columns+'S'
###################################################
DATA_PATH = '.PreviousSeasonRankings'
pattern = '20[0-9][0-9]-20[0-9][0-9]'
datasetP = merge_data(DATA_PATH, pattern)
Y2 = actual_result(datasetP)
datasetP = datasetP[columns]
datasetP.columns = datasetP.columns+'P'
dataset = pd.concat([datasetS, datasetP], axis=1)

###################################################
###################################################
#5a) Feature Exploration
###################################################
###################################################
'''
#Describe
describe = dataset.describe().T
describe = describe.round(2)
describe.set_index = dataset.columns
print(describe)
#describe.to_csv('Results/Describe.csv')
#dependent variable
unique, counts = np.unique(Y, return_counts=True)
print(unique, counts)
#correlation matrix
#print(dataset.columns)
plt.matshow(dataset.corr(), interpolation='nearest', cmap = 'RdYlGn')
plt.title('Heatmap', y=1.01)
plt.gca().xaxis.tick_bottom()
plt.xlabel('column number')
plt.ylabel('column number')
plt.colorbar()
plt.show()
'''
###################################################
###################################################
#Late Addition Dec 2021) Remodelling dataset to containg double the number of features
###################################################
###################################################
#home/away becomes H/A
datasetHA = copy.deepcopy(dataset)
datasetHA.columns = [column_name.replace('home', 'H').replace("away","A") for column_name in datasetHA]
datasetHA["IsHomeTeamH"] = 1
#home/away becomes A/H
datasetAH = copy.deepcopy(dataset)
datasetAH.columns = [column_name.replace('home', 'A').replace("away","H") for column_name in datasetAH]
datasetAH["IsHomeTeamH"] = 0

dataset = datasetHA.append(datasetAH)
Y = np.append(Y,1-Y) #flip result for features where home and away is swapped
###################################################
###################################################
#5b) Feature Extraction--all results are saved to csv andbest models incoperated into modelling
###################################################
###################################################
SelectKBest_ = SelectKBest_(dataset, Y)
columns = ['Columns', 'Feature Count', 'Mean', 'SD']
SelectKBest_ = pd.DataFrame.from_records(SelectKBest_, columns = columns)
SelectKBest_.to_csv('Results/SelectKBest.csv', index = False)
'''
print(len(dataset))
#5ba) logistic regression brute force
print('logistic regression brute force')
brute_force_lr = brute_force_lr(dataset, Y)
columns = ['Columns', 'Feature Count', 'Mean', 'SD']
brute_force_lr = pd.DataFrame.from_records(brute_force_lr, columns = columns)
brute_force_lr.to_csv('Results/brute_force_lr.csv', index = False)

#5bb) SelectKBest
print('SelectKBest')
SelectKBest_ = SelectKBest_(dataset, Y)
columns = ['Columns', 'Feature Count', 'Mean', 'SD']
SelectKBest_ = pd.DataFrame.from_records(SelectKBest_, columns = columns)
SelectKBest_.to_csv('Results/SelectKBest.csv', index = False)

#5bc) SequentialForwardSelection
print('SequentialForwardSelection')
SequentialForwardSelection = SequentialForwardSelection(dataset, Y)
columns = ['Columns', 'Feature Count', 'Mean', 'SD']
SequentialForwardSelection = pd.DataFrame.from_records(SequentialForwardSelection, columns = columns)
SequentialForwardSelection.to_csv('Results/SequentialForwardSelection.csv', index = False)

#5bd) SequentialBackwardSelection
print('SequentialBackwardSelection')
SequentialBackwardSelection = SequentialBackwardSelection(dataset, Y)
columns = ['Columns', 'Feature Count', 'Mean', 'SD']
SequentialBackwardSelection = pd.DataFrame.from_records(SequentialBackwardSelection, columns = columns)
SequentialBackwardSelection.to_csv('Results/SequentialBackwardSelection.csv', index = False)

#5be) PCA
print('PCA')
PCA_ = PCA_(dataset, Y)
columns = ['Feature Count', 'Mean', 'SD']
PCA_ = pd.DataFrame.from_records(PCA_, columns = columns)
PCA_.to_csv('Results/PCA_.csv', index = False)
'''
###################################################
###################################################
#Final Model
###################################################
###################################################
'''
#save season and ewma data as a csv file
DATA_PATH = 'Data/1 Season'
pattern = '20[0-9][0-9]-20[0-9][0-9]'
datasetF = merge_data(DATA_PATH, pattern)
datasetF = load_data(datasetF, '')
#get seasons
season = datasetF.Season
#get load from overtime data and save new columns to csv
ewma = return_ewma_column(datasetF, 7)
ewma = return_ewma_column(datasetF, 28)
ewma = back2back(datasetF)
ewma.to_csv('Data/AllEWMAandSeasons.csv', index = False)


emea_and_season_data = pd.read_csv('Data/AllEWMAandSeasons.csv')
dataset['home_ewma7'] = emea_and_season_data.home_ewma7
dataset['away_ewma7'] = emea_and_season_data.away_ewma7
dataset['home_ewma28'] = emea_and_season_data.home_ewma28
dataset['away_ewma28'] = emea_and_season_data.away_ewma28
dataset['home_ratio'] = dataset['home_ewma7'] / dataset['home_ewma28']
dataset['away_ratio'] = dataset['away_ewma7'] / dataset['away_ewma28']
#back to back was explored lagte but didnt improve results
#dataset['home_b2b'] = emea_and_season_data.home_back2back
#dataset['away_b2b'] = emea_and_season_data.away_back2back
dataset['Y'] = Y
dataset['season'] = emea_and_season_data.Season


#print(len(dataset.columns))
Previous3Seasons = pd.DataFrame(MasterModel(dataset, '3PreviousSeasons')).T
Previous3Seasons.columns = ['2007-2008', '2008-2009', '2009-2010',
                            '2010-2011', '2011-2012', '2012-2013',
                            '2013-2014', '2014-2015', '2015-2016',
                            '2016-2017', '2017-2018', '2018-2019',
                            '2019-2020']
Previous3Seasons['Av'] = Previous3Seasons.mean(axis = 1)
Previous3Seasons = Previous3Seasons.sort_values('Av', ascending = False)
#Previous3Seasons.to_csv('Results/Eval3PreviousSeasons.csv')
print(Previous3Seasons.head())

AllPreviousSeasons = pd.DataFrame(MasterModel(dataset, 'AllPreviousSeasons')).T
AllPreviousSeasons.columns = [             '2006-2007', '2007-2008',
                              '2008-2009', '2009-2010', '2010-2011',
                              '2011-2012', '2012-2013', '2013-2014',
                              '2014-2015', '2015-2016', '2016-2017',
                              '2017-2018', '2018-2019', '2019-2020']
AllPreviousSeasons['Av'] = AllPreviousSeasons.mean(axis = 1)
AllPreviousSeasons = AllPreviousSeasons.sort_values('Av', ascending = False)
#AllPreviousSeasons.to_csv('Results/EvalAllPreviousSeasonsWithoutFirst.csv')
print(AllPreviousSeasons.head())

#rolling averages plot

rolling_avg = data_to_plot(dataset, '3PreviousSeasons')
rolling_avg.to_csv('Results/RollingAvg3PreviousSeasons.csv')

rolling_avg = data_to_plot(dataset, 'AllPreviousSeasons')
rolling_avg.to_csv('Results/RollingAvgAllPreviousSeasons.csv')

###################################################
###################################################
#Surprise analysis
###################################################
###################################################
availability_analysis(dataset, 10, '3PreviousSeasons')
availability_analysis(dataset, 10, 'AllPreviousSeasons')
'''