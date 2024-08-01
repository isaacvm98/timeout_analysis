import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplay,leaguegamefinder,leaguedashteamstats
from nba_api.stats.static import teams
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
pd.options.display.max_columns = None

SEASONS = ['2019','2020','2021','2022','2023']
teams_ids = [x['id'] for x in teams.get_teams()]
all_games_df = pd.DataFrame()
for team in teams_ids:
    games = leaguegamefinder.LeagueGameFinder(team_id_nullable=team).get_data_frames()[0]
    games['SEASON_YEAR'] = games['SEASON_ID'].str[-4:]
    games = games[games['SEASON_YEAR'].isin(SEASONS)]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    all_games_df = pd.concat([all_games_df,games],ignore_index=True)
teams_name_df = all_games_df[(all_games_df['MATCHUP'].str.contains('vs'))][['GAME_ID']]
teams_name_df['HomeName'] = all_games_df[(all_games_df['MATCHUP'].str.contains('vs'))]['MATCHUP'].str.split('vs.',expand=True)[0]
teams_name_df['AwayName'] = all_games_df[(all_games_df['MATCHUP'].str.contains('vs'))]['MATCHUP'].str.split('vs.',expand=True)[1]

NBA_TEAMS = ['DEN', 'LAC', 'CHA', 'ORL', 'IND', 'NOP', 'UTA', 'TOR', 'MIA','SAC', 'POR', 'BKN', 'GSW', 'LAL', 'PHI', 'MIL', 'CHI', 'HOU','MEM', 'DET', 'DAL', 'CLE', 'MIN', 'OKC', 'PHX', 'NYK',
     'SAS', 'WAS', 'ATL','BOS']

pbp = pd.read_pickle(r'C:\Users\isaac\Desktop\Proyectos\nba_analysis\timeout_analysis\preprocessing\data\pbp_2019-2023.pkl')
pbp = pbp.merge(teams_name_df,on='GAME_ID')
pbp['GAME_ID_INT'] = pbp['GAME_ID'].astype(int)
pbp = pbp.sort_values(['GAME_ID','EVENTNUM'])
pbp = pbp[(pbp['AwayName'].isin(NBA_TEAMS))&(pbp['HomeName'].isin(NBA_TEAMS))]
turnover_keywords = ['BLOCK', 'STEAL']
# Function to determine possession
def determine_possession(row):
    if row['VISITORDESCRIPTION'] and row['HOMEDESCRIPTION']:
        # If both descriptions are present, determine possession based on turnover keywords
        if any(keyword in row['VISITORDESCRIPTION'] for keyword in turnover_keywords):
            return row['HomeName']
        elif any(keyword in row['HOMEDESCRIPTION'] for keyword in turnover_keywords):
            return row['AwayName']
    # Default case where only one of the descriptions is present
    return row['AwayName'] if row['VISITORDESCRIPTION'] else row['HomeName']
def coalesce(*args):
    return next((arg for arg in args if pd.notnull(arg)), np.nan)
# Define a helper function to get the last non-NaN value
def last_non_nan(series):
    return series.dropna().iloc[-1] if not series.dropna().empty else np.nan
# Define the getmode function
def getmode(x):
    uniqx, counts = np.unique(x, return_counts=True)
    return uniqx[np.argmax(counts)]

# Helper function to find first occurrence index
def first_occurrence(series):
    return (series.index[0] if series.any() else np.inf)

# Define a function to determine the first timeout used during a run
def first_timeout(x, timeout_col):
    timeouts = x[x[timeout_col]].index  # Get indices where timeout is True
    return timeouts[0] if not timeouts.empty else np.inf

# Define a function to determine period change within a run
def period_change(x):
    changes = x.index[x.shift(-1) != x]  # Get indices where period changes
    return changes[0] if not changes.empty else np.inf


# Convert game clock display to seconds remaining
pbp['qtr_seconds_remaining'] = pbp['PCTIMESTRING'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
# Data formatted so that events attributed to possessing team
# Remove non-action events
pbp = pbp[(pbp['NEUTRALDESCRIPTION'].isnull())]
pbp = pbp[~pbp['VISITORDESCRIPTION'].fillna('').str.contains("SUB")]
pbp = pbp[~pbp['VISITORDESCRIPTION'].fillna('').str.contains('REPLAY')]
pbp = pbp[~pbp['VISITORDESCRIPTION'].fillna('').str.contains("KICKED")]
pbp = pbp[~pbp['VISITORDESCRIPTION'].fillna('').str.contains("FOUL")]
pbp = pbp[~pbp['HOMEDESCRIPTION'].fillna('').str.contains("SUB")]
pbp = pbp[~pbp['HOMEDESCRIPTION'].fillna('').str.contains('REPLAY')]
pbp = pbp[~pbp['HOMEDESCRIPTION'].fillna('').str.contains("KICKED")]
pbp = pbp[~pbp['HOMEDESCRIPTION'].fillna('').str.contains("FOUL")]
# Note upcoming timeouts before removing them
pbp['home_timeout_after'] = pbp['HOMEDESCRIPTION'].shift(-1).fillna('').str.contains("Timeout")
pbp['away_timeout_after'] = pbp['VISITORDESCRIPTION'].shift(-1).fillna('').str.contains("Timeout")
pbp = pbp[~pbp['VISITORDESCRIPTION'].fillna('').str.contains("Timeout") & ~pbp['HOMEDESCRIPTION'].fillna('').str.contains("Timeout")]
pbp['possession'] = pbp.apply(determine_possession, axis=1)

# # Group by GameID and possession
pbp['possession'] = pbp.groupby(['GAME_ID', 'possession'])['possession'].transform(getmode)

# # Assign key column to clusters of data
pbp['possession_id'] = pbp.groupby('GAME_ID')['possession'].apply(lambda x: (x != x.shift()).cumsum()+1).values
pbp[['VISITORSCORE','HOMESCORE']] = pbp['SCORE'].str.split('-',expand=True)
pbp['HOMESCORE'] = pbp['HOMESCORE'].astype(str).str.replace('None','NaN').astype(float)
pbp['VISITORSCORE'] = pbp['VISITORSCORE'].astype(str).str.replace('None','NaN').astype(float)
#pbp['HOMESCORE'].fillna(method='ffill',inplace=True)
#pbp['VISITORSCORE'].fillna(method='ffill',inplace=True)
pbp['coalesce_event'] = pbp.apply(lambda row: coalesce(row['VISITORDESCRIPTION'], row['HOMEDESCRIPTION']), axis=1)

# Group by the relevant columns and summarize each possession
possession_summary = pbp.groupby(['GAME_ID', 'PERIOD', 'HomeName', 'AwayName', 'possession_id','possession']).agg(
    start=('qtr_seconds_remaining', 'min'),
    end=('qtr_seconds_remaining', 'max'),
    end_home_score=('HOMESCORE', 'max'),
    end_away_score=('VISITORSCORE', 'max'),
    home_timeout_after=('home_timeout_after',lambda x: True if True in x.value_counts() else False),
    away_timeout_after=('away_timeout_after',lambda x: True if True in x.value_counts() else False),
    last_event=('coalesce_event', 'last'),
).reset_index()
possession_summary['end_home_score'].fillna(0,inplace=True)
possession_summary['end_away_score'].fillna(0,inplace=True)

# Keep track of score
possession_summary['score'] = np.where(possession_summary['possession'] == possession_summary['HomeName'],
                                       possession_summary['end_home_score'] > possession_summary['end_home_score'].shift(),
                                       possession_summary['end_away_score'] > possession_summary['end_away_score'].shift())
possession_summary['last_score'] = np.where(possession_summary['score'], possession_summary['possession'], np.nan)

# Keep track of who scored last
possession_summary['last_score'] = possession_summary['last_score'].ffill()
possession_summary['last_score'] = possession_summary['last_score'].shift()

# Team identities required for matchup modeling
possession_summary['home_poss'] = possession_summary['possession'] == possession_summary['HomeName']
possession_summary['offense'] = np.where(possession_summary['home_poss'], possession_summary['HomeName'], possession_summary['AwayName'])
possession_summary['defense'] = np.where(possession_summary['home_poss'], possession_summary['AwayName'], possession_summary['HomeName'])
possession_summary['gp'] = possession_summary['GAME_ID'].astype(str) + possession_summary['PERIOD'].astype(str)
possession_summary = possession_summary[(possession_summary['end_home_score']>0)|(possession_summary['end_away_score']>0)]

possession_summary['run_id'] = possession_summary.groupby(['GAME_ID'])['offense'].apply(lambda x: (x != x.shift()).cumsum()).values
possession_summary['run_id_gp'] = possession_summary.groupby(['gp'])['offense'].apply(lambda x: (x != x.shift()).cumsum()).values
possession_summary.reset_index(inplace=True,drop=True)
possession_summary['to_start'] = possession_summary['start'] - 1


runs = possession_summary.groupby(['GAME_ID','gp', 'run_id_gp']).agg(
    home_team = ('HomeName','last'),
    away_team_team = ('AwayName','last'),
    run_length = ('last_event','size'),
    #score = ('score',lambda x:True if True in x.to_list() else False),
    run_team = ('possession',lambda x:getmode(x)),
    previous_run_team = ('defense',lambda x:getmode(x)),
    home_score = ('end_home_score','max'),
    away_score = ('end_away_score','max'),
    home_to = ('home_timeout_after', lambda x: 1 if True in x.to_list() else 0),
    away_to = ('away_timeout_after', lambda x: 1 if True in x.to_list() else 0),
    home_to_index = ('home_timeout_after', lambda x: x.to_list().index(True) if True in x.to_list() else 0),
    away_to_index = ('away_timeout_after', lambda x: x.to_list().index(True) if True in x.to_list() else 0),
    start_time = ('end','max'),
    end_time = ('start','min')
).reset_index()
runs['next_run_length'] = runs.groupby(['GAME_ID', 'gp'])['run_length'].shift(-1).fillna(0).astype(int)
runs['next_home_score'] = runs.groupby(['GAME_ID', 'gp'])['home_score'].shift(-1).fillna(0).astype(int)
runs['next_away_score'] = runs.groupby(['GAME_ID', 'gp'])['away_score'].shift(-1).fillna(0).astype(int)


runs['previous_run_length'] = runs.groupby(['GAME_ID', 'gp'])['run_length'].shift(1).fillna(0).astype(int)
runs['previous_home_score'] = runs.groupby(['GAME_ID', 'gp'])['home_score'].shift(1).fillna(0).astype(int)
runs['previous_away_score'] = runs.groupby(['GAME_ID', 'gp'])['away_score'].shift(1).fillna(0).astype(int)
runs['home_net_diff'] = runs['home_score'] - runs['away_score']
runs['away_net_diff'] = runs['away_score'] - runs['home_score']
runs['run_time']  = runs['start_time'] - runs.groupby(['GAME_ID', 'gp'])['start_time'].shift(-1).fillna(0).astype(int)
dict_all_runs = {
    'avg_opponent_run_length': [],
    'opponent_run_length_max': [],
    'opponent_run_length_min': [],
    'avg_own_run_length': [],
    'own_run_length_max': [],
    'own_run_length_min': [],
}
for ix,group in tqdm(runs.groupby(['GAME_ID','gp', 'run_id_gp'])):
    run_id = group['run_id_gp'].iloc[0] 
    run_team = group['run_team'].iloc[0]
    gp = group['gp'].iloc[0]
    previous_5_runs_own = [run_id-i for i in range(2, 14,2)]
    previous_5_runs_opponent = [run_id-i for i in range(1, 12,2)]
    
    # Get the previous 5 runs by the same team and opponent
    prev_own_runs = runs[(runs['gp'] == gp) & (runs['run_team'] == run_team) & (runs['run_id_gp'].isin(previous_5_runs_own))]
    prev_opponent_runs = runs[(runs['gp'] == gp) & (runs['run_team'] != run_team) & (runs['run_id_gp'].isin(previous_5_runs_opponent))]

    if not prev_own_runs.empty:
        prev_runs = prev_own_runs['run_length']
        avg_own_run_length = prev_runs.mean()
        own_run_length_max = prev_runs.max()
        own_run_length_min = prev_runs.min()
    else:
        avg_own_run_length, own_run_length_max, own_run_length_min = None, None, None

    if not prev_opponent_runs.empty:
        prev_runs = prev_opponent_runs['run_length']
        avg_opponent_run_length = prev_runs.mean()
        opponent_run_length_max = prev_runs.max()
        opponent_run_length_min = prev_runs.min()
    else:
        avg_opponent_run_length, opponent_run_length_max, opponent_run_length_min = None, None, None
        
    dict_all_runs['avg_opponent_run_length'].append(avg_opponent_run_length)
    dict_all_runs['opponent_run_length_max'].append(opponent_run_length_max)
    dict_all_runs['opponent_run_length_min'].append(opponent_run_length_min)
    dict_all_runs['avg_own_run_length'].append(avg_own_run_length)
    dict_all_runs['own_run_length_max'].append(own_run_length_max)
    dict_all_runs['own_run_length_min'].append(own_run_length_min)

runs_stats = pd.DataFrame(dict_all_runs)

runs = pd.concat([runs,runs_stats],axis=1)

runs.to_pickle('runs_19-23.pkl')
