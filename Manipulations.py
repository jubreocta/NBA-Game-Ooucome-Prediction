import pandas as pd
import os
import re
def home_away_different_teams(dataset):
    '''creates different team name tags for the home and away teams in a dataset
    '''
    dataset['Home Team'] = 'Home ' + dataset['Home Team']
    dataset['Away Team'] = 'Away ' + dataset['Away Team']
    return dataset

def merge_data(DATA_PATH, pattern):
    '''merges all files ina folder into one dataset'''
    pattern = re.compile(pattern)
    rank_votings = dict()
    dataset = pd.DataFrame()
    for file in os.listdir(DATA_PATH):
        if pattern.match(file):
            import_ = pd.read_csv(DATA_PATH + '/' + file)
            dataset = dataset.append(import_)
    dataset.reset_index(inplace = True)
    return dataset

def two_seasons_merge(DATA_PATH, pattern):
    '''used in the special case of previous season ranking.
    merges 2 file in each 2 seasons folder in a directory
    '''
    pattern = re.compile(pattern)
    rank_votings = dict()
    dataset = pd.DataFrame()
    for file in os.listdir(DATA_PATH):
        if pattern.match(file):
            import_ = pd.read_csv(DATA_PATH + '/' + file)
            dataset = dataset.append(import_)
    dataset.reset_index(inplace = True)
    return dataset
