import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplay,leaguegamefinder,leaguedashteamstats
from nba_api.stats.static import teams
from time import sleep
from tqdm import tqdm
DATA_PATH = r'C:\Users\isaac\Desktop\Proyectos\nba_analysis\timeout_analysis\preprocessing\data'
# Define a helper function to handle coalesce logic
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

SEASONS = ['2019','2020','2021','2022','2023']
teams_ids = [x['id'] for x in teams.get_teams()]
all_games_df = pd.DataFrame()
for team in teams_ids:
    games = leaguegamefinder.LeagueGameFinder(team_id_nullable=team).get_data_frames()[0]
    games['SEASON_YEAR'] = games['SEASON_ID'].str[-4:]
    games = games[games['SEASON_YEAR'].isin(SEASONS)]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    all_games_df = pd.concat([all_games_df,games],ignore_index=True)

# df = pd.DataFrame()
# for game in tqdm(all_games_df['GAME_ID'].unique()):
#     pbp = playbyplay.PlayByPlay(game_id=game).get_data_frames()[0]
#     df = pd.concat([df,pbp],ignore_index=True)
#     sleep(1)
# df.to_pickle(DATA_PATH+f'\pbp_{SEASONS[0]}-{SEASONS[-1]}.pkl')

print('Finished downloading play-by-play data')

# # Define keywords for turnovers
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

teams_name_df = all_games_df[(all_games_df['MATCHUP'].str.contains('vs'))][['GAME_ID']]
teams_name_df['HomeName'] = all_games_df[(all_games_df['MATCHUP'].str.contains('vs'))]['MATCHUP'].str.split('vs.',expand=True)[0]
teams_name_df['AwayName'] = all_games_df[(all_games_df['MATCHUP'].str.contains('vs'))]['MATCHUP'].str.split('vs.',expand=True)[1]

df = pd.read_pickle(DATA_PATH+'\pbp_2019-2023.pkl')
df = df.merge(teams_name_df,on='GAME_ID')
# Turn play-by-play data into possessions
pbp = df.copy()

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
#pbp = pbp[~pbp['VISITORDESCRIPTION'].fillna('').str.contains("Timeout") & ~pbp['HOMEDESCRIPTION'].fillna('').str.contains("Timeout")]
pbp['possession'] = pbp.apply(determine_possession, axis=1)

# # Group by GameID and possession
pbp['possession'] = pbp.groupby(['GAME_ID', 'possession'])['possession'].transform(getmode)

# # Assign key column to clusters of data
pbp['possession_id'] = pbp.groupby('GAME_ID')['possession'].apply(lambda x: (x != x.shift()).cumsum()+1).values
pbp[['VISITORSCORE','HOMESCORE']] = pbp['SCORE'].str.split('-',expand=True)
pbp['HOMESCORE'] = pbp['HOMESCORE'].astype(str).str.replace('None','NaN').astype(float)
pbp['VISITORSCORE'] = pbp['VISITORSCORE'].astype(str).str.replace('None','NaN').astype(float)
# pbp['HOMESCORE'].fillna(method='ffill',inplace=True)
# pbp['VISITORSCORE'].fillna(method='ffill',inplace=True)
pbp['coalesce_event'] = pbp.apply(lambda row: coalesce(row['VISITORDESCRIPTION'], row['HOMEDESCRIPTION']), axis=1)
# # Summary info for each possession

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
leaguegamefinder.LeagueGameFinder(game_id_nullable=possession_summary['GAME_ID'].iloc[0])
leaguegamefinder.LeagueGameFinder(game_id_nullable=possession_summary['GAME_ID'].iloc[-1])
possession_summary.to_pickle(DATA_PATH+f'\possession_summary_{SEASONS[0]}-{SEASONS[-1]}.pkl')
print('Finished processing possessions')

runs = pd.read_pickle(DATA_PATH+f'\possession_summary_{SEASONS[0]}-{SEASONS[-1]}.pkl')

runs['run_id'] = runs.groupby(['GAME_ID'])['offense'].apply(lambda x: (x != x.shift()).cumsum()+1).values
runs['run_id_gp'] = runs.groupby(['gp'])['offense'].apply(lambda x: (x != x.shift()).cumsum()+1).values
runs.reset_index(inplace=True,drop=True)
runs['to_start'] = runs['start'] - 1

dict_all_runs = {
    'run_id_gp': [],
    'gp': [],
    'home_team': [],
    'away_team': [],
    'home_score': [],
    'away_score': [],
    'run_length': [],
    'to': [],
    'home_to': [],
    'away_to': [],
    'to_index': [],
    'changed_run': [],
    'run_team': [],
    'next_run_length': [],
    'next_run_home_score': [],
    'next_run_away_score': [],
    'next_run_team': [],
    'previous_run_length': [],
    'previous_run_home_score': [],
    'previous_run_away_score': [],
    'previous_run_team': [],
    'avg_opponent_run_length': [],
    'opponent_run_length_max': [],
    'opponent_run_length_min': [],
    'avg_own_run_length': [],
    'own_run_length_max': [],
    'own_run_length_min': [],
    'to_team': []
}

for ix, group in tqdm(runs.groupby(['gp', 'run_id_gp'])):
    run_id = group['run_id_gp'].iloc[0]
    gp = group['gp'].iloc[0]
    home_team = group['HomeName'].iloc[0]
    away_team = group['AwayName'].iloc[0]
    home_score = group['end_home_score'].iloc[-1]
    away_score = group['end_away_score'].iloc[-1]
    run_length = group.shape[0]
    run_team = group['possession'].transform(getmode).iloc[0]
    previous_run_team = group['defense'].transform(getmode).iloc[0]
    gp_int = int(gp)
    # Calculate timeouts and run changes
    if True in group['home_timeout_after'].values:
        to = 1
        home_to = 1
        away_to = 0
        to_list = group['home_timeout_after'].to_list()
        true_index = to_list.index(True)
        if true_index + 1 == run_length:
            changed_run = 1
        timeout = 'home'
    elif True in group['away_timeout_after'].values:
        to = 1
        home_to = 0
        away_to = 1
        to_list = group['away_timeout_after'].to_list()
        true_index = to_list.index(True)
        if true_index + 1 == run_length:
            changed_run = 1
        timeout = 'away'
    else:
        to, home_to, away_to, true_index, changed_run = 0, 0, 0, 0, 0
        timeout = np.nan

    # Find next and previous runs within the same game-period (gp)
    next_run = runs[(runs['gp'] == gp) & (runs['run_id_gp'] == run_id + 1)]
    prev_run = runs[(runs['gp'] == gp) & (runs['run_id_gp'] == run_id - 1)]

    if not next_run.empty:
        next_run_length = next_run.shape[0]
        next_run_home_score = next_run['end_home_score'].iloc[-1]
        next_run_away_score = next_run['end_away_score'].iloc[-1]
        next_run_team = next_run['offense'].iloc[0]
    else:
        try:
            next_gp = str(gp_int + 1).zfill(11)
            next_run = runs[(runs['gp'] == next_gp)]
            next_run = next_run[next_run['run_id_gp'] == next_run['run_id_gp'].min()]
            next_run_length = next_run.shape[0]
            next_run_home_score = next_run['end_home_score'].iloc[-1]
            next_run_away_score = next_run['end_away_score'].iloc[-1]
            next_run_team = next_run['offense'].iloc[0]
        except:
            next_run_length, next_run_home_score, next_run_away_score, next_run_team = 0, 0, 0, np.nan

    if not prev_run.empty:
        previous_run_length = prev_run.shape[0]
        previous_run_home_score = prev_run['end_home_score'].iloc[-1]
        previous_run_away_score = prev_run['end_away_score'].iloc[-1]
        previous_run_team = prev_run['offense'].iloc[0]
    else:
        try:
            prev_gp = str(gp_int - 1).zfill(11)
            previous_run = runs[(runs['gp'] == prev_gp)]
            previous_run = previous_run[previous_run['run_id_gp'] == previous_run['run_id_gp'].max()]
            previous_run_length = previous_run.shape[0]
            previous_run_home_score = previous_run['end_home_score'].iloc[-1]
            previous_run_away_score = previous_run['end_away_score'].iloc[-1]
            previous_run_team = previous_run['offense'].iloc[0]
        except:
            previous_run_length, previous_run_home_score, previous_run_away_score, previous_run_team = 0, 0, 0, np.nan
    previous_5_runs_own = [run_id-i for i in range(2, 14,2)]
    previous_5_runs_opponent = [run_id-i for i in range(1, 12,2)]
    # Get the previous 5 runs by the same team and opponent
    prev_own_runs = runs[(runs['gp'] == gp) & (runs['offense'] == run_team) & (runs['run_id_gp'].isin(previous_5_runs_own))]
    prev_opponent_runs = runs[(runs['gp'] == gp) & (runs['offense'] != run_team) & (runs['run_id_gp'].isin(previous_5_runs_opponent))]

    if not prev_own_runs.empty:
        avg_own_run_length = (prev_own_runs.groupby('run_id_gp').size()).mean()
        own_run_length_max = (prev_own_runs.groupby('run_id_gp').size()).max()
        own_run_length_min = (prev_own_runs.groupby('run_id_gp').size()).min()
    else:
        avg_own_run_length, own_run_length_max, own_run_length_min = None, None, None

    if not prev_opponent_runs.empty:
        avg_opponent_run_length = (prev_opponent_runs.groupby('run_id_gp').size()).mean()
        opponent_run_length_max = (prev_opponent_runs.groupby('run_id_gp').size()).max()
        opponent_run_length_min = (prev_opponent_runs.groupby('run_id_gp').size()).min()
    else:
        avg_opponent_run_length, opponent_run_length_max, opponent_run_length_min = None, None, None
    if timeout == 'home':
        to_team = home_team
    elif timeout == 'away':
        to_team = away_team
    else:
        to_team = np.nan
    # Append values to the dictionary during your analysis
    dict_all_runs['run_id_gp'].append(run_id)
    dict_all_runs['gp'].append(gp)
    dict_all_runs['home_team'].append(home_team)
    dict_all_runs['away_team'].append(away_team)
    dict_all_runs['home_score'].append(home_score)
    dict_all_runs['away_score'].append(away_score)
    dict_all_runs['run_team'].append(run_team)
    dict_all_runs['run_length'].append(run_length)
    dict_all_runs['to'].append(to)
    dict_all_runs['home_to'].append(home_to)
    dict_all_runs['away_to'].append(away_to)
    dict_all_runs['to_index'].append(true_index)
    dict_all_runs['changed_run'].append(changed_run)
    dict_all_runs['next_run_length'].append(next_run_length)
    dict_all_runs['next_run_home_score'].append(next_run_home_score)
    dict_all_runs['next_run_away_score'].append(next_run_away_score)
    dict_all_runs['next_run_team'].append(next_run_team)
    dict_all_runs['previous_run_length'].append(previous_run_length)
    dict_all_runs['previous_run_home_score'].append(previous_run_home_score)
    dict_all_runs['previous_run_away_score'].append(previous_run_away_score)
    dict_all_runs['previous_run_team'].append(previous_run_team)
    dict_all_runs['avg_opponent_run_length'].append(avg_opponent_run_length)
    dict_all_runs['opponent_run_length_max'].append(opponent_run_length_max)
    dict_all_runs['opponent_run_length_min'].append(opponent_run_length_min)
    dict_all_runs['avg_own_run_length'].append(avg_own_run_length)
    dict_all_runs['own_run_length_max'].append(own_run_length_max)
    dict_all_runs['own_run_length_min'].append(own_run_length_min)
    dict_all_runs['to_team'].append(to_team)

all_runs_to = pd.DataFrame(dict_all_runs)
for ix, row in all_runs_to.iterrows():
    if row['to']==1:
        if row['home_to']==1:
            all_runs_to.loc[ix,'to_team'] = row['home_team']
        else:
            all_runs_to.loc[ix,'to_team'] = row['away_team']
    else:
        all_runs_to.loc[ix,'to_team'] = np.nan

for ix, row in all_runs_to.iterrows():
    if row['to']==1:
        if row['run_team'] != row['to_team']:
            all_runs_to.loc[ix,'to_non_run'] = 1
        else:
            all_runs_to.loc[ix,'to_non_run'] = 0
    else:
        all_runs_to.loc[ix,'to_non_run'] = np.nan

to_sec_start_df = runs[(runs['home_timeout_after']==True)|(runs['away_timeout_after']==True)].groupby(['gp','run_id_gp'],as_index=False)['to_start'].last()
dict_times = {'previous_time_list' : [],
'min_time_list' : [],
'time_spent_list' : [],
'to_time_list ': []}
for ix,group in tqdm(runs.groupby(['gp','run_id_gp'])):
    min_time = group['start'].min()
    gp = group['gp'].iloc[0]
    run_id = group['run_id_gp'].iloc[0]
    if run_id == 2:
        previous_run_min_time = 720
        time_spent = previous_run_min_time - min_time
    else:
        previous_run_min_time = runs[(runs['gp']==gp)&(runs['run_id_gp']==run_id-1)]['start'].min()
        time_spent = previous_run_min_time - min_time
    df_sec_to = to_sec_start_df.loc[(to_sec_start_df['gp']==gp)&(to_sec_start_df['run_id_gp']==run_id)]
    if df_sec_to.empty:
        to_time = np.nan
    else:
        to_time = df_sec_to['to_start'].iloc[0]
    dict_times['previous_time_list'].append(previous_run_min_time)
    dict_times['min_time_list'].append(min_time)
    dict_times['time_spent_list'].append(time_spent)
    dict_times['to_time_list '].append(to_time)
times_df = pd.DataFrame(dict_times)
times_df.columns = ['previous_run_time','last_run_time','time_spent_run','to_time']

all_runs_to = pd.concat([all_runs_to,times_df],axis=1)

for ix, row in all_runs_to.iterrows():
    if row['to']==1:
        if row['home_to']==1:
            all_runs_to.loc[ix,'to_team'] = row['home_team']
        else:
            all_runs_to.loc[ix,'to_team'] = row['away_team']
    else:
        all_runs_to.loc[ix,'to_team'] = np.nan

for ix, row in all_runs_to.iterrows():
    if row['to']==1:
        if row['run_team'] != row['to_team']:
            all_runs_to.loc[ix,'to_non_run'] = 1
        else:
            all_runs_to.loc[ix,'to_non_run'] = 0
    else:
        all_runs_to.loc[ix,'to_non_run'] = np.nan

to_sec_start_df = runs[(runs['home_timeout_after']==True)|(runs['away_timeout_after']==True)].groupby(['gp','run_id_gp'],as_index=False)['to_start'].last()
dict_times = {'previous_time_list' : [],
'min_time_list' : [],
'time_spent_list' : [],
'to_time_list ': []}
for ix,group in tqdm(runs.groupby(['gp','run_id_gp'])):
    min_time = group['start'].min()
    run_id = group['run_id_gp'].iloc[0]
    if run_id == 2:
        previous_run_min_time = 720
        time_spent = previous_run_min_time - min_time
    else:
        previous_run_min_time = runs[(runs['gp']==gp)&(runs['run_id_gp']==run_id-1)]['start'].min()
        time_spent = previous_run_min_time - min_time
    df_sec_to = to_sec_start_df.loc[(to_sec_start_df['gp']==gp)&(to_sec_start_df['run_id_gp']==run_id)]
    if df_sec_to.empty:
        to_time = np.nan
    else:
        to_time = df_sec_to['to_start'].iloc[0]
    dict_times['previous_time_list'].append(previous_run_min_time)
    dict_times['min_time_list'].append(min_time)
    dict_times['time_spent_list'].append(time_spent)
    dict_times['to_time_list '].append(to_time)
times_df = pd.DataFrame(dict_times)
times_df.columns = ['previous_run_time','last_run_time','time_spent_run','to_time']

all_runs_to = pd.concat([all_runs_to,times_df],axis=1)

all_runs_to.to_pickle(f'all_runs_to_{SEASONS[0]}-{SEASONS[-1]}.pkl')

print('Finished processing runs')