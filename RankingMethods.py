import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
def team_order(dataset):
    '''creates a dictionary with teams as items and row number as values to create
    an ordering of teams each season so that matrices and vectors are alligned'''
    teams = sorted(set(dataset['Home Team']).union(set(dataset['Away Team'])))
    return {key: value for value,key in enumerate(teams)}


def points_for(full_data, dataset_of_interest, n):
    '''Creates a vector containing the total number of points scored by each
    team over the time span of the supplied dataset.'''
    team_ordering = team_order(full_data)
    array = np.array([0] * n)
    for i in range(len(dataset_of_interest)):
        #add value of point scored to array
        array[team_ordering[dataset_of_interest['Home Team'][i]]] += dataset_of_interest['Home Score'][i]
        array[team_ordering[dataset_of_interest['Away Team'][i]]] += dataset_of_interest['Away Score'][i]
    return array

        
def points_against(full_data, dataset_of_interest, n):
    '''Creates a vector containing the total number of points conceded by each
    team over the time span of the supplied dataset'''
    team_ordering = team_order(full_data)
    array = np.array([0] * n)
    for i in range(len(dataset_of_interest)):
        #add value of point conceeded to array
        array[team_ordering[dataset_of_interest['Home Team'][i]]] += dataset_of_interest['Away Score'][i]
        array[team_ordering[dataset_of_interest['Away Team'][i]]] += dataset_of_interest['Home Score'][i]
    return array


def total_no_played(full_data, dataset_of_interest, n):
    '''Creates a diagonal matrix containing the total number of games played
    by each team over the time span of the supplied dataset'''
    team_ordering = team_order(full_data)
    array = np.array([[0]*n]*n)
    for i in range(len(dataset_of_interest)):
        #add 1 for each game played
        home_index = team_ordering[dataset_of_interest['Home Team'][i]]
        away_index = team_ordering[dataset_of_interest['Away Team'][i]]
        array[home_index, home_index] += 1
        array[away_index, away_index] += 1
    return array


def total_no_won(full_data, dataset_of_interest, n):
    '''Creates a diagonal matrix containing the total number of games won
    by each team over the time span of the supplied dataset'''
    team_ordering = team_order(full_data)
    array = np.array([[0]*n]*n)
    for i in range(len(dataset_of_interest)):
        #add 1 for each game won
        if dataset_of_interest['Home Score'][i] > dataset_of_interest['Away Score'][i]:
            home_index = team_ordering[dataset_of_interest['Home Team'][i]]
            array[home_index, home_index] += 1
        else:
            away_index = team_ordering[dataset_of_interest['Away Team'][i]]
            array[away_index, away_index] += 1
    return array


def pairwise_matchups(full_data, dataset_of_interest, n):
    '''Creates an off-diagonal matrix containing the number of pairwise
    matchups between teams over the time span of the supplied dataset'''
    team_ordering = team_order(full_data)
    array = np.array([[0]*n]*n)
    for i in range(len(dataset_of_interest)):
        #add 1 for each game between distinct oponents
        home_index = team_ordering[dataset_of_interest['Home Team'][i]]
        away_index = team_ordering[dataset_of_interest['Away Team'][i]]
        array[home_index, away_index] += 1
        array[away_index, home_index] += 1
    return array


def points_given_up(full_data, dataset_of_interest, n, type_):
    '''Creates a matrix containing the total number of points given up to each
    team over the time span of the supplied dataset. The types represent the
    different forms of voting described in the textbook.'''
    team_ordering = team_order(full_data)
    matrix = np.array([[0]*n]*n)
    for i in range(len(dataset_of_interest)):
        #add value of point conceeded to array
        home_index = team_ordering[dataset_of_interest['Home Team'][i]]
        away_index = team_ordering[dataset_of_interest['Away Team'][i]]
        if type_ == 3:
            matrix[home_index, away_index] += dataset_of_interest['Away Score'][i]
            matrix[away_index, home_index] += dataset_of_interest['Home Score'][i]
        elif type_ == 2:
            if dataset_of_interest['Home Score'][i] < dataset_of_interest['Away Score'][i]:
                matrix[home_index, away_index] += dataset_of_interest['Away Score'][i] - dataset_of_interest['Home Score'][i]
            else:
                matrix[away_index, home_index] += dataset_of_interest['Home Score'][i] - dataset_of_interest['Away Score'][i]
        elif type_ == 1:
            if dataset_of_interest['Home Score'][i] < dataset_of_interest['Away Score'][i]:
                matrix[home_index, away_index] += 1
            else:
                matrix[away_index, home_index] += 1
    return matrix

def subtract_losses_from_wins(full_data, dataset_of_interest, n):
    '''Creates a vector containing the total number of losses subtracted from
    the total number of wins for each team over the time span of the supplied
    dataset'''
    team_ordering = team_order(full_data)
    array = np.array([0]*n)
    for i in range(len(dataset_of_interest)):
        home_index = team_ordering[dataset_of_interest['Home Team'][i]]
        away_index = team_ordering[dataset_of_interest['Away Team'][i]]
        #checks who won the game
        if dataset_of_interest['Home Score'][i] > dataset_of_interest['Away Score'][i]:
            array[home_index] += 1
            array[away_index] -= 1
        else:
            array[home_index] -= 1
            array[away_index] += 1     
    return array



def win_percentage(dataset):
    '''Computes the win percentage of each team. This function works over a
    sorted dataframe by continuously updating the matrices over the span of
    a supplied dataset.'''
    team_ordering = team_order(dataset)
    n = len(team_ordering)
    home_rating   = []
    away_rating   = []
    unique_dates = dataset.Date.unique()
    for date_index in range(len(unique_dates)):
        data_used_for_ranking = dataset[dataset.Date == unique_dates[date_index-1]]
        data_used_for_ranking.reset_index(inplace = True)
        data_to_be_ranked = dataset[dataset.Date == unique_dates[date_index]]
        data_to_be_ranked.reset_index(inplace = True)
        if date_index == 0:
            total_no_won_array = np.array([[0]*n]*n)
            total_no_played_array = np.array([[0]*n]*n)
        else:
            total_no_won_array   += total_no_won(dataset, data_used_for_ranking, n)
            total_no_played_array   += total_no_played(dataset, data_used_for_ranking, n)            
        #r =  np.where(total_no_played_array.diagonal() != 0,
        #              np.divide(total_no_won_array.diagonal(), total_no_played_array.diagonal()), 0)
        #this longer method is necessary to avoid division by 0
        r = []
        for i in range(len(total_no_played_array.diagonal())):
            if total_no_played_array.diagonal()[i] == 0:
                r.append(0)
            else:
                r.append(total_no_won_array.diagonal()[i] / total_no_played_array.diagonal()[i])
        r = np.array(r)    
        #append win percentage
        for game_index in range(len(data_to_be_ranked)):
            home_index = team_ordering[data_to_be_ranked['Home Team'][game_index]]
            home_rating.append(r[home_index])
            away_index = team_ordering[data_to_be_ranked['Away Team'][game_index]]
            away_rating.append(r[away_index])
    dataset['home_win_percentage'] = home_rating
    dataset['away_win_percentage'] = away_rating
    return dataset


def massey_for_a_season(dataset):
    '''Impements the massey ranking method by solving masseys matrix equation
    Mr = p. This function works over a sorted dataframe by continuously
    updating the matrices over the span of a supplied dataset.'''
    team_ordering = team_order(dataset)
    n = len(team_ordering)
    home_overall_rating   = []
    home_offensive_rating = []
    home_defensive_rating = []
    away_overall_rating   = []
    away_offensive_rating = []
    away_defensive_rating = []
    
    unique_dates = dataset.Date.unique()
    for date_index in range(len(unique_dates)):
        data_used_for_ranking = dataset[dataset.Date == unique_dates[date_index-1]]
        data_used_for_ranking.reset_index(inplace = True)
        data_to_be_ranked = dataset[dataset.Date == unique_dates[date_index]]
        data_to_be_ranked.reset_index(inplace = True)       

        if date_index == 0:
            points_for_array = np.array([0]*n)
            points_against_array = np.array([0]*n)
            pairwise_matchups_array = np.array([[0]*n]*n)
            total_no_played_array = np.array([[0]*n]*n)
            
        else:
            total_no_played_array   += total_no_played(dataset, data_used_for_ranking, n)
            pairwise_matchups_array += pairwise_matchups(dataset, data_used_for_ranking, n)
            points_for_array        += points_for(dataset, data_used_for_ranking, n)
            points_against_array    += points_against(dataset, data_used_for_ranking, n)
        #Massy rating calculations start here
        T = total_no_played_array
        P = pairwise_matchups_array
        f = points_for_array
        a = points_against_array
        p = f - a
        M = T - P
        #letters are used inline with book to make referencing easy
        M[n-1,:] = 1
        p[n-1] = 0
        try:
            r = np.linalg.solve(M,p)
            d = np.linalg.solve(T+P,np.dot(T,r) - f)
            o = r - d            
        except np.linalg.LinAlgError:
            d,o,r = ([[None]*n]*3)
        #Placing ranks of teams in the dataset
        for game_index in range(len(data_to_be_ranked)):
            home_index = team_ordering[data_to_be_ranked['Home Team'][game_index]]
            home_overall_rating.append(r[home_index])
            home_offensive_rating.append(o[home_index])
            home_defensive_rating.append(d[home_index])
            away_index = team_ordering[data_to_be_ranked['Away Team'][game_index]]
            away_overall_rating.append(r[away_index])
            away_offensive_rating.append(o[away_index])
            away_defensive_rating.append(d[away_index])
    dataset['massey_home_overall_rating']   = home_overall_rating
    dataset['massey_home_offensive_rating'] = home_offensive_rating
    dataset['massey_home_defensive_rating'] = home_defensive_rating
    dataset['massey_away_overall_rating']   = away_overall_rating
    dataset['massey_away_offensive_rating'] = away_offensive_rating
    dataset['massey_away_defensive_rating'] = away_defensive_rating
    return dataset


def colley_for_a_season(dataset):
    '''Impements the colley ranking method by solving colleys matrix equation
    Cr = b. This function works over a sorted dataframe by continuously
    updating the matrices over the span of a supplied dataset.'''
    team_ordering = team_order(dataset)
    n = len(team_ordering)
    home_rating   = []
    away_rating   = []
    
    unique_dates = dataset.Date.unique()
    for date_index in range(len(unique_dates)):
        data_used_for_ranking = dataset[dataset.Date == unique_dates[date_index-1]]
        data_used_for_ranking.reset_index(inplace = True)
        data_to_be_ranked = dataset[dataset.Date == unique_dates[date_index]]
        data_to_be_ranked.reset_index(inplace = True)
        if date_index == 0:
            subtract_losses_from_wins_array = np.array([0]*n)
            pairwise_matchups_array = np.array([[0]*n]*n)
            total_no_played_array = np.array([[0]*n]*n)
            np.fill_diagonal(total_no_played_array, 2)
        else:
            total_no_played_array   += total_no_played(dataset, data_used_for_ranking, n)
            pairwise_matchups_array -= pairwise_matchups(dataset, data_used_for_ranking, n)
            subtract_losses_from_wins_array += subtract_losses_from_wins(dataset, data_used_for_ranking, n)
        #Colley rating calculations start here
        T = total_no_played_array
        P = pairwise_matchups_array
        C = T + P
        b = 1 + 0.5 * subtract_losses_from_wins_array
        #letters are used inline with book to make referencing easy
        try:
            r = np.linalg.solve(C,b)
        except np.linalg.LinAlgError:
            r = ([None]*n)
        #Placing ranks of teams in the dataset)
        for game_index in range(len(data_to_be_ranked)):
            home_index = team_ordering[data_to_be_ranked['Home Team'][game_index]]
            home_rating.append(r[home_index])
            away_index = team_ordering[data_to_be_ranked['Away Team'][game_index]]
            away_rating.append(r[away_index])
    dataset['colley_home_rating'] = home_rating
    dataset['colley_away_rating'] = away_rating
    return dataset


def markov_for_a_season(dataset, type_ = 3, beta = 0.6):
    '''Impements the markov ranking method by solving for the stationary
    vector of a voting (winners and losers voting points) stochastic matrix.
    This function works over a sorted dataframe by continuously updating the
    matrices over the span of a supplied dataset.
    types:
    1 -- loser votes only one point for winner
    2 -- loser votes point differential
    3 -- both winner and looser vote points given up
    '''
    team_ordering = team_order(dataset)
    n = len(team_ordering)
    home_rating = []
    away_rating = []
    
    unique_dates = dataset.Date.unique()
    for date_index in range(len(unique_dates)):
        data_used_for_ranking = dataset[dataset.Date == unique_dates[date_index-1]]
        data_used_for_ranking.reset_index(inplace = True)
        data_to_be_ranked = dataset[dataset.Date == unique_dates[date_index]]
        data_to_be_ranked.reset_index(inplace = True)       

        if date_index == 0:
            voting_matrix = np.array([[0]*n]*n)
            
        else:
            voting_matrix += points_given_up(dataset, data_used_for_ranking, n, type_)
        #create stocastic matrix
        #line below uses equal voting to other teams by team that has not lost
        #for loop uses vote to self there reorganises the matrix.
        S = np.nan_to_num(voting_matrix/voting_matrix.sum(axis=1, keepdims=True), nan = 1/n)
        for i in range(len(S)):
            if (S[i] == np.array([1/n]*n)).all():
                S[i] = np.array([0]*n)
                S[i][i] = 1
        #calculate the stationary vector
        S = beta * S + (1 - beta) / n * np.array([[1]*n]*n)
        A=np.append(np.transpose(S)-np.identity(n),[[1]*n],axis=0)
        b=np.transpose(np.append(np.array([0]*n),1))
        try:
            r = np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b))
        except np.linalg.LinAlgError:
            r = ([None]*n)
        #Placing ranks of teams in the dataset)
        for game_index in range(len(data_to_be_ranked)):
            home_index = team_ordering[data_to_be_ranked['Home Team'][game_index]]
            home_rating.append(r[home_index])
            away_index = team_ordering[data_to_be_ranked['Away Team'][game_index]]
            away_rating.append(r[away_index])
    dataset['markov_home_rating'+str(type_)] = home_rating
    dataset['markov_away_rating'+str(type_)] = away_rating
    return dataset



def od_for_a_season(dataset):
    '''Impements the offense-defence ranking method by solving for the stationary
    vector of a voting (winners and losers voting points) stochastic matrix.
    This function works over a sorted dataframe by continuously updating the
    matrices over the span of a supplied dataset.
    '''
    team_ordering = team_order(dataset)
    n = len(team_ordering)
    home_overall_rating   = []
    home_offensive_rating = []
    home_defensive_rating = []
    away_overall_rating   = []
    away_offensive_rating = []
    away_defensive_rating = []
    
    unique_dates = dataset.Date.unique()
    for date_index in range(len(unique_dates)):
        data_used_for_ranking = dataset[dataset.Date == unique_dates[date_index-1]]
        data_used_for_ranking.reset_index(inplace = True)
        data_to_be_ranked = dataset[dataset.Date == unique_dates[date_index]]
        data_to_be_ranked.reset_index(inplace = True)       

        if date_index == 0:
            voting_matrix = np.array([[0]*n]*n)
            
        else:
            voting_matrix += points_given_up(dataset, data_used_for_ranking, n, 3)

        A = voting_matrix
        d = np.array([1.0]*n).reshape(n,1)
        old_d = np.array([0.9]*n).reshape(n,1)
        k = 1
        while k < 10 and np.allclose(old_d,d) is False: #k used to be less than 1000 but this produces rankings that are too high. extreme outlier
            old_d = d
            o = np.transpose(A).dot(np.reciprocal(old_d))
            d = A.dot(np.reciprocal(o))
            k += 1
        d,o,r = d,o,o/d
        #Placing ranks of teams in the dataset
        for game_index in range(len(data_to_be_ranked)):
            home_index = team_ordering[data_to_be_ranked['Home Team'][game_index]]
            home_overall_rating.append(r[home_index][0])
            home_offensive_rating.append(o[home_index][0])
            home_defensive_rating.append(d[home_index][0])
            away_index = team_ordering[data_to_be_ranked['Away Team'][game_index]]
            away_overall_rating.append(r[away_index][0])
            away_offensive_rating.append(o[away_index][0])
            away_defensive_rating.append(d[away_index][0])
    dataset['od_home_overall_rating']   = home_overall_rating
    dataset['od_home_offensive_rating'] = home_offensive_rating
    dataset['od_home_defensive_rating'] = home_defensive_rating
    dataset['od_away_overall_rating']   = away_overall_rating
    dataset['od_away_offensive_rating'] = away_offensive_rating
    dataset['od_away_defensive_rating'] = away_defensive_rating
    return dataset


def rating_for_less(dataset, months, ranking, beta = 0.6):
    '''Impements the different ranking methods over a shorter time span dependent on the
    value of n supplied. n represents how many months (30 days) backward we want to look at
    match results to determine what the team ratings are going to be.
    '''
    unique_dates = dataset.Date.unique()
    if ranking == 'colley' or 'markov' in ranking:
        home_rating   = []
        away_rating   = []
    elif ranking == 'massey' or ranking == 'od':
            home_overall_rating   = []
            home_offensive_rating = []
            home_defensive_rating = []
            away_overall_rating   = []
            away_offensive_rating = []
            away_defensive_rating = []
        
    for date_index in range(len(unique_dates)):
        end   = pd.to_datetime(unique_dates[date_index])
        start = pd.to_datetime(unique_dates[date_index] - np.timedelta64(30*months,'D'))
        data = dataset.loc[(dataset['Date']>= start)]
        data = data.loc[(data['Date']<= end)]
        if ranking == 'colley':
            colley_for_a_season(data)
            data = data[data['Date']== unique_dates[date_index]]
            data.reset_index(inplace = True)
            for row in range(len(data)):
                home_rating.append(data.loc[row, 'colley_home_rating'])
                away_rating.append(data.loc[row, 'colley_away_rating'])
        elif  'markov' in ranking:
            type_ = int(ranking[-1])
            markov_for_a_season(data, type_ = type_, beta = beta)
            data = data[data['Date']== unique_dates[date_index]]
            data.reset_index(inplace = True)
            for row in range(len(data)):
                home_rating.append(data.loc[row, 'markov_home_rating'+str(type_)])
                away_rating.append(data.loc[row, 'markov_away_rating'+str(type_)])
        elif ranking == 'massey':
            massey_for_a_season(data)
            data = data[data['Date']== unique_dates[date_index]]
            data.reset_index(inplace = True)
            for row in range(len(data)):
                home_overall_rating.append(data.loc[row, 'massey_home_overall_rating'])
                home_offensive_rating.append(data.loc[row, 'massey_home_offensive_rating'])
                home_defensive_rating.append(data.loc[row, 'massey_home_defensive_rating'])
                away_overall_rating.append(data.loc[row, 'massey_away_overall_rating'])
                away_offensive_rating.append(data.loc[row, 'massey_away_offensive_rating'])
                away_defensive_rating.append(data.loc[row, 'massey_away_defensive_rating'])
        elif ranking == 'od':
            od_for_a_season(data)
            data = data[data['Date']== unique_dates[date_index]]
            data.reset_index(inplace = True)
            for row in range(len(data)):
                home_overall_rating.append(data.loc[row, 'od_home_overall_rating'])
                home_offensive_rating.append(data.loc[row, 'od_home_offensive_rating'])
                home_defensive_rating.append(data.loc[row, 'od_home_defensive_rating'])
                away_overall_rating.append(data.loc[row, 'od_away_overall_rating'])
                away_offensive_rating.append(data.loc[row, 'od_away_offensive_rating'])
                away_defensive_rating.append(data.loc[row, 'od_away_defensive_rating'])
    if ranking == 'colley':
        dataset['colley_home_%d_month' % months] = home_rating
        dataset['colley_away_%d_month' % months] = away_rating
    elif 'markov' in ranking:
        dataset[ranking + '_home_%d_month' % months] = home_rating
        dataset[ranking + '_away_%d_month' % months] = away_rating
    elif ranking == 'massey':
        dataset['massey_home_overall_%d_month' % months] = home_overall_rating
        dataset['massey_away_overall_%d_month' % months] = away_overall_rating
        dataset['massey_home_offensive_%d_month' % months] = home_offensive_rating
        dataset['massey_away_offensive_%d_month' % months] = away_offensive_rating
        dataset['massey_home_defensive_%d_month' % months] = home_defensive_rating
        dataset['massey_away_defensive_%d_month' % months] = away_defensive_rating
    elif ranking == 'od':
        dataset['od_home_overall_%d_month' % months] = home_overall_rating
        dataset['od_away_overall_%d_month' % months] = away_overall_rating
        dataset['od_home_offensive_%d_month' % months] = home_offensive_rating
        dataset['od_away_offensive_%d_month' % months] = away_offensive_rating
        dataset['od_home_defensive_%d_month' % months] = home_defensive_rating
        dataset['od_away_defensive_%d_month' % months] = away_defensive_rating
        
    return dataset

def do_seasonal_ranking(data, type_ = 'All'):
    '''takes in a dataset and spits out rankings for every home team and away
    team. This by default doest all types of rankings for a season'''
    year = max(data.Date).year
    percentage = win_percentage(data)
    #print(percentage)
    colley = colley_for_a_season(data)
    #print(colley)
    massey = massey_for_a_season(data)
    #print(massey)
    od = od_for_a_season(data)
    #print(od)
    markov1 = markov_for_a_season(data,1)
    #print(markov1)
    markov2 = markov_for_a_season(data,2)
    #print(markov2)
    markov3 = markov_for_a_season(data,3)
    #print(markov3)
    return markov3

def do_timed_ranking(data, type_ = 'All'):
    '''takes in a dataset and spits out rankings for every home team and away
    team. This by default doest all types of rankings for different month lengths
    from 1 month previous to 7 months previous'''
    year = max(data.Date).year
    for i in range(1,8):
        print( 'month', i)
        colley = rating_for_less(data, i, 'colley')
        massey = rating_for_less(data, i, 'massey')
        markov1 = rating_for_less(data, i, 'markov1')
        markov2 = rating_for_less(data, i, 'markov2')
        markov3 = rating_for_less(data, i, 'markov3')
        od = rating_for_less(data, i, 'od')
    #od.to_csv('Rankings/' + type_ + ' ' + str(year-1) + '-' + str(year) + '.csv')
    return od
