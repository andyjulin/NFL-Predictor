import requests

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup


def get_stats(season, week):
    # Set included datasets to parse
    cols = ['passing', 'rushing', 'kicking', 'punting', 'returning', 'defense', 'downs', 'yardage', 'turnovers']
    
    # Grab each value, and remove blank values (-)
    df = pd.concat([get_category_stats(season, week, category) for category in cols], axis = 1).replace('-', '0')
        
    # Replace % based strings with floats
    for s in ['Completion %', 'Field Goal %', 'Extra Point %', 'Third Down %', 'Fourth Down %']:
        df[s] = df[s].apply(lambda x: float(x.strip('%')) / 100.0)
    
    # Convert any other strings to numeric values
    df = df.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))

    return df

def get_category_stats(season, week, category):
    # Set website to grab data from (Fox Sports was the only one that kept weekly data...)
    request = 'season=' + str(season) + '&week=' + str(100 + week) + '&category=' + str(category).upper()
    url = 'http://www.foxsports.com/nfl/team-stats?' + request
    
    # Some extra tables for later categories?  Not sure why...
    offset = 0 if category in ['passing', 'rushing', 'kicking', 'punting'] else 2
    
    # Load HTML into array
    html = BeautifulSoup(requests.get(url).text, 'lxml')    
    table = html.find_all('table')[0]
    header = table.find_all('tr')[offset]
    rows = table.find_all('tr')[offset + 1:]
    
    # Set Column Names
    ths = header.find_all('th')
    cols = [category.title() + ' Rank'] + [t['title'] for t in ths[1:]]
    
    # Get inner HTML values
    tds = [r.find_all('td') for r in rows]
    vals = [[v.get_text() for v in t] for t in tds]

    # Remove excess whitespace around values
    N = [v[0].strip('\t\n\r').split('\n') for v in vals]
    N = filter(lambda a: a != [''], N)

    # Create NumPy arrays
    ranks = np.array([n[0] for n in N])
    codes = np.array([n[3] for n in N])
        
    # Remove more blank values
    V = [v[1:] for v in vals]
    V = filter(lambda a: a != [''], V)

    # Apparently the 2001 - Week 1 table is screwed up with blank entries...
    if (season == 2001 and week == 2):
        ranks = ranks[:28]
        codes = codes[:28]
        V     = V[:28]
                
    # Create pandas DataFrame
    D = np.concatenate((np.array([ranks]).T, np.array(V)), axis = 1)    

    return pd.DataFrame(D, index = codes, columns = cols)



def get_results(season, week):
    url = 'http://www.nfl.com/scores/%d/REG%d#' % (season, week)

    # Load HTML into array
    html = BeautifulSoup(requests.get(url).text, 'lxml')  
    divs = html.find_all('div', { 'class' : 'new-score-box' })
    p_names = [d.find_all('p', { 'class' : 'team-name'})[i] for i in [0, 1] for d in divs]
    p_records = [d.find_all('p', { 'class' : 'team-record'})[i] for i in [0, 1] for d in divs]
    p_scores  = [d.find_all('p', { 'class' : 'total-score'})[i] for i in [0, 1] for d in divs]
        
    # Grab game data from arrays
    teams = [p.a['href'].replace('/teams/profile?team=', '') for p in p_names]
    teams = [t.replace('JAC', 'JAX').replace('ARI', 'ARZ').replace('WAS', 'WSH') for t in teams] # Argh...
    N = len(teams)
    
    opponents = [teams[(i + N / 2) if (i < N / 2) else (i - N / 2)] for i in range(N)]
    records = [p.get_text().strip().replace('(', '').replace(')', '').split('-') for p in p_records]
    scores  = [p.get_text().strip().replace('--', '-1') for p in p_scores]
                                
    # Calculate winners from game data        
    results = [[int(r) for r in records[i]] + [int(i >= N / 2)] + [opponents[i]] + [int(scores[i])] for i in range(N)]    
    wins = [int(results[(i + N / 2)][5] < results[i][5]) for i in range(N / 2)]   
    victories = wins + [int(not w) for w in wins]    
    
    data = [[teams[i]] + results[i] + [victories[i]] for i in range(N)]
        
    # Combine into DataFrame
    return pd.DataFrame(data, columns = ['Team', 'Wins', 'Losses', 'Ties', 'Home', 'Opponent', 'Points', 'Victory'], index = teams)      

