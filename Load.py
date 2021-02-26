def load_data(filename, how = 'from_file'):
    '''loads raw a dataseet and transforms it from the raw form to columns
    understandable by other parts of the code if the data set has not been
    transformed. It could load a raw file (how = 'from_file') or a dataset
    already within the program. It also renames all nba teams where name
    changes have occured so that all previous seasons tally with the newest
    name of the team
    '''
    import pandas as pd
    if how == 'from_file':
        data = pd.read_csv(filename)#, index_col = 0)
    else:
        data = filename
    data[['Date','Time']] = data['Date'].str.split(" ",expand=True,)
    try:
        data[['Home Score','Away Score']] = data.Result.str.split(" - ",expand=True,)
    except:
        data[['Home Score','Away Score']] = (None,None)

    data['Home Team'] = data['Home Team'].replace({'New Jersey Nets':'Brooklyn Nets',
                                                'New Orleans Hornets': 'New Orleans Pelicans',
                                                'Seattle SuperSonics':'Oklahoma City Thunder',
                                                'New Orleans/Oklahoma City Hornets':'New Orleans Pelicans',
                                                'Charlotte Bobcats': 'Charlotte Hornets'})
    data['Away Team'] = data['Away Team'].replace({'New Jersey Nets':'Brooklyn Nets',
                                                'New Orleans Hornets': 'New Orleans Pelicans',
                                                'Seattle SuperSonics':'Oklahoma City Thunder',
                                                'New Orleans/Oklahoma City Hornets':'New Orleans Pelicans',
                                                'Charlotte Bobcats': 'Charlotte Hornets'})
    data['Date']       = pd.to_datetime(data['Date'], format = '%d/%m/%Y')
    data['Home Score'] = pd.to_numeric(data['Home Score'])
    data['Away Score'] = pd.to_numeric(data['Away Score'])
    #data = data.dropna()
    data.reset_index(inplace = True, drop = True)
    #print(data.head())
    #print(data.dtypes)
    #print(len(data))
    return data

