'''used to tidy raw files downloaded from basketball reference'''
import os
import re
import datetime
import pandas as pd
DATA_PATH = 'Data/raw from BR'
pattern = re.compile(r'^20[0-9][0-9]-20[0-9][0-9]')
for file in os.listdir(DATA_PATH):
    #print(file[:-4])
    if pattern.match(file):
        dataset = pd.read_csv(DATA_PATH + "/" + file)
        dataset['Dates'] = dataset['Date']+ ' ' + dataset['Start (ET)'] + 'M'
        dataset['Dates'] = dataset['Dates'].map(lambda x: datetime.datetime.strptime(x, '%a %b %d %Y %H:%M%p').strftime('%d/%m/%Y %H:%M'))
        dataset['Results'] = dataset['PTS.1'].astype('str') + ' - ' + dataset['PTS'].astype('str')
        dataset['overtime'] = dataset['Unnamed: 7']
        dataset = dataset[['Dates', 'Home/Neutral', 'Visitor/Neutral', 'Results', 'overtime']]
        dataset.columns = ['Date', 'Home Team', 'Away Team', 'Result', 'Overtime']
        dataset['Season'] = file[:-4]
        dataset.to_csv('Data/1 Season/' + file, index = False, encoding = "utf-8")

print(dataset)
