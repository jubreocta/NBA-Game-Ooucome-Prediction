'''We consider all previous work as generalised learning on what we want our final model to look like.
First thing we have finalised is our method of evaluation. We would stick to using 3 previous seasons to predict 1.
We would use this method to cross validate and select our final features and then test this on an LSTM'''
import pandas as pd
#pd.set_option('expand_frame_repr', False)
#pd.set_option('display.max_rows', 500)
#pd.set_option('display.max_columns', 500)
from Evaluation import actual_result
from Manipulations import merge_data
'''Step 1) Combine all useful data into one dataset'''

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

emea_and_season_data = pd.read_csv('Data/AllEWMAandSeasons.csv')
dataset['home_ewma7'] = emea_and_season_data.home_ewma7
dataset['away_ewma7'] = emea_and_season_data.away_ewma7
dataset['home_ewma28'] = emea_and_season_data.home_ewma28
dataset['away_ewma28'] = emea_and_season_data.away_ewma28
dataset['home_ratio'] = dataset['home_ewma7'] / dataset['home_ewma28']
dataset['away_ratio'] = dataset['away_ewma7'] / dataset['away_ewma28']
dataset['Y'] = Y
dataset['season'] = emea_and_season_data.Season

print(dataset.tail())