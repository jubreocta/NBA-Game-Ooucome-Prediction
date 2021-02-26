import pandas as pd
def ewma(dataset, days):
    '''returns a dictionary of keys as teams and values as the teams laod over a seasons stretch'''
    dataset['Overtime'] = dataset['Overtime'].fillna('0')
    dataset['Overtime'] = dataset['Overtime'].replace({'OT':'1OT'})
    load_dict = dict() #stores dictionary of team as items and dataframe with loading per day as values
    lambda_ = 2 / (days + 1)
    maxdate = max(dataset['Date'])
    mindate = min(dataset['Date'])
    calendar = pd.date_range(start=mindate, end=maxdate)
    teams = sorted(set(dataset['Home Team']).union(set(dataset['Away Team'])))
    for team in teams:
        all_team_data = []
        for date_index in range(len(calendar)):
            row = []
            row.append(calendar[date_index])
            #grab from full dataset only fixtures from that date to check if team played
            dates_data = dataset[dataset.Date == calendar[date_index]]
            dates_data.reset_index(inplace = True)
            #iterate through this mini dataset
            if len(dates_data) == 0:
                loading = 0
            else:
                for index in range(len(dates_data)):
                    if dates_data.loc[index, 'Home Team'] == team or dates_data.loc[index, 'Away Team'] == team:
                        #load value is the number of minutes played
                        if 'OT' in dates_data.loc[index, 'Overtime']:
                            #loading is 48 + number of overtime periods times 5
                            loading = 48 + int(dates_data.loc[index, 'Overtime'][0]) * 5
                        else:
                            loading = 48                        
                        break
                    else:
                        loading = 0
            row.append(loading)
            all_team_data.append(row)
        load_dataset = pd.DataFrame.from_records(data = all_team_data,columns = ['Date','Load'])
        # calculate the ewma
        ewma = []
        ewma_yesterday = 0
        for index in range(len(load_dataset)):
            ewma_today = load_dataset.loc[index, 'Load'] * lambda_ + ((1 - lambda_) * ewma_yesterday)
            ewma.append(ewma_today)
            ewma_yesterday = ewma_today
        load_dataset['ewma'+ str(days)] = ewma
        load_dict[team] = load_dataset
    return load_dict

def return_ewma_column(dataset, days):
    '''returns a dataset with additional columns that show the ewma load calculated
    based on a number of days supplied to the function'''
    load_dict = ewma(dataset, days) #calls the other function to get a dictionary
    home_load   = []
    away_load   = []
    for index in range(len(dataset)):
        #ifetch datasets from the dictionary
        home_team_load_dataset = load_dict[dataset.loc[index, 'Home Team']]
        away_team_load_dataset = load_dict[dataset.loc[index, 'Away Team']]
        date      = dataset.loc[index, 'Date']
        
        try:
            h_load = home_team_load_dataset.loc[home_team_load_dataset['Date'] == date - pd.DateOffset(1), 'ewma'+str(days)].iloc[0]        
            a_load = away_team_load_dataset.loc[away_team_load_dataset['Date'] == date - pd.DateOffset(1), 'ewma'+str(days)].iloc[0]        
            home_load.append(h_load)
            away_load.append(a_load)
        except:
            home_load.append(0)
            away_load.append(0)            
    dataset['home_ewma' + str(days)] = home_load
    dataset['away_ewma' + str(days)] = away_load
    return dataset



def back2back(dataset):
    '''adds boolean columns marking whether or not a team is playing a back
    to back game'''
    home_info = []
    away_info = []
    for index in range(len(dataset)):
        date = dataset.loc[index, 'Date']
        home_team = dataset.loc[index, 'Home Team']
        away_team = dataset.loc[index, 'Away Team']
        yday = date - pd.DateOffset(1)
        yday_data = dataset.where(dataset.Date == yday)
        yday_teams = set(yday_data['Home Team']).union(set(yday_data['Away Team']))
        if home_team in yday_teams:
            home_info.append(1)
        else:
            home_info.append(0)
        if away_team in yday_teams:
            away_info.append(1)
        else:
            away_info.append(0)
    dataset['home_back2back'] = home_info
    dataset['away_back2back'] = away_info
    return dataset
