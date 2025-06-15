import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DB_PATH = 'nba.sqlite'

# Past champions data
past_champions = {
    '22004': 1610612759,  # Spurs
    '22005': 1610612748,  # Heat
    '22006': 1610612759,  # Spurs
    '22007': 1610612738,  # Celtics
    '22008': 1610612747,  # Lakers
    '22009': 1610612747,  # Lakers
    '22010': 1610612742,  # Mavericks
    '22011': 1610612748,  # Heat  
    '22013': 1610612759,  # Spurs
    '22014': 1610612744,  # Warriors
    '22015': 1610612739,  # Cavaliers
    '22016': 1610612744,  # Warriors
    '22017': 1610612744,  # Warriors
    '22018': 1610612761,  # Raptors
    '22019': 1610612747,  # Lakers
    '22020': 1610612749,  # Bucks
    '22021': 1610612744,  # Warriors 
    '22022': 1610612743   # Nuggets
}

def get_team_choices():
    """Fetch team choices from the database."""
    conn = sqlite3.connect(DB_PATH)
    df_teams = pd.read_sql_query("SELECT id, full_name FROM team;", conn)
    conn.close()
    team_choices = {int(row['id']): row['full_name'] for _, row in df_teams.iterrows()}
    return team_choices

def fetch_games(team_id):
    """Fetch games relevant to the given team_id, excluding the 2003 and 2012 seasons."""
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT * FROM game
    WHERE (team_id_home = {team_id} OR team_id_away = {team_id})
    AND ((season_id >= 22004 AND season_id <= 22022 AND season_id != 22012) 
    OR (season_id >= 42004 AND season_id <= 42022 AND season_id != 42012))
    """
    df_games = pd.read_sql_query(query, conn)
    conn.close()
    return df_games

def is_champ(past_champions, team_id, season_id):
    """Check if the chosen team is the champion of the given season."""
    season_id = int(season_id)
    return past_champions.get(str(season_id)) == team_id

def calculate_advanced_stats(df_games, team_id):
    """Calculate advanced statistics for the selected team for each regular season."""
    team_id = int(team_id)
    df_games = df_games.astype({'team_id_home': 'int32', 'team_id_away': 'int32'})
    
    reg_years = sorted(df_games[df_games['season_type'] == 'Regular Season']['season_id'].unique())
    
    season_stats = {}
    
    for season_id in reg_years:
        reg_games = df_games[(df_games['season_id'] == season_id) & (df_games['season_type'] == 'Regular Season')]
        
        # Get games for this specific team
        home_games = reg_games[reg_games['team_id_home'] == team_id]
        away_games = reg_games[reg_games['team_id_away'] == team_id]
        
        # Calculate total games the team actually played
        team_total_games = len(home_games) + len(away_games)

        if team_total_games == 0:
            # Initialize with zeros if no games
            season_stats[season_id] = {
                'reg_win_pct': 0,
                'avg_point_diff': 0,
                'fg_pct': 0,
                'fg3_pct': 0,
                'ft_pct': 0,
                'reb_per_game': 0,
                'ast_per_game': 0,
                'tov_per_game': 0,
                'stl_per_game': 0,
                'blk_per_game': 0,
                'fg3a_per_game': 0
            }
            continue

        # Calculate wins and win percentage
        home_wins = home_games[home_games['pts_home'] > home_games['pts_away']].shape[0]
        away_wins = away_games[away_games['pts_away'] > away_games['pts_home']].shape[0]
        reg_win_percentage = (home_wins + away_wins) / team_total_games

        # Calculate point differential
        home_point_diff = (home_games['pts_home'] - home_games['pts_away']).sum()
        away_point_diff = (away_games['pts_away'] - away_games['pts_home']).sum()
        avg_point_diff = (home_point_diff + away_point_diff) / team_total_games

        # Calculate shooting percentages and stats
        # For home games (team is home)
        home_stats = {
            'fgm': home_games['fgm_home'].sum(),
            'fga': home_games['fga_home'].sum(),
            'fg3m': home_games['fg3m_home'].sum(),
            'fg3a': home_games['fg3a_home'].sum(),
            'ftm': home_games['ftm_home'].sum(),
            'fta': home_games['fta_home'].sum(),
            'reb': home_games['reb_home'].sum(),
            'ast': home_games['ast_home'].sum(),
            'tov': home_games['tov_home'].sum(),
            'stl': home_games['stl_home'].sum(),
            'blk': home_games['blk_home'].sum()
        }
        
        # For away games (team is away)
        away_stats = {
            'fgm': away_games['fgm_away'].sum(),
            'fga': away_games['fga_away'].sum(),
            'fg3m': away_games['fg3m_away'].sum(),
            'fg3a': away_games['fg3a_away'].sum(),
            'ftm': away_games['ftm_away'].sum(),
            'fta': away_games['fta_away'].sum(),
            'reb': away_games['reb_away'].sum(),
            'ast': away_games['ast_away'].sum(),
            'tov': away_games['tov_away'].sum(),
            'stl': away_games['stl_away'].sum(),
            'blk': away_games['blk_away'].sum()
        }

        # Combine home and away stats
        total_stats = {key: home_stats[key] + away_stats[key] for key in home_stats.keys()}

        # Calculate percentages (avoid division by zero)
        fg_pct = total_stats['fgm'] / total_stats['fga'] if total_stats['fga'] > 0 else 0
        fg3_pct = total_stats['fg3m'] / total_stats['fg3a'] if total_stats['fg3a'] > 0 else 0
        ft_pct = total_stats['ftm'] / total_stats['fta'] if total_stats['fta'] > 0 else 0

        # Calculate per-game averages
        reb_per_game = total_stats['reb'] / team_total_games
        ast_per_game = total_stats['ast'] / team_total_games
        tov_per_game = total_stats['tov'] / team_total_games
        stl_per_game = total_stats['stl'] / team_total_games
        blk_per_game = total_stats['blk'] / team_total_games
        fg3a_per_game = total_stats['fg3a'] / team_total_games

        season_stats[season_id] = {
            'reg_win_pct': reg_win_percentage,
            'avg_point_diff': avg_point_diff,
            'fg_pct': fg_pct,
            'fg3_pct': fg3_pct,
            'ft_pct': ft_pct,
            'reb_per_game': reb_per_game,
            'ast_per_game': ast_per_game,
            'tov_per_game': tov_per_game,
            'stl_per_game': stl_per_game,
            'blk_per_game': blk_per_game,
            'fg3a_per_game': fg3a_per_game
        }

    return season_stats

def data_cleaning(team_id, past_champions):
    """Clean and prepare the data for the selected team with advanced stats."""
    data_frame = fetch_games(team_id)
    
    season_stats = calculate_advanced_stats(data_frame, team_id)
    
    # Convert keys to int
    season_stats = {int(k): v for k, v in season_stats.items()}

    # Only use seasons where the team actually played (has win percentage data)
    data = []
    for season_id, stats in season_stats.items():
        if stats['reg_win_pct'] > 0 or season_id in map(int, past_champions.keys()):
            champ_status = is_champ(past_champions, team_id, season_id)

            row = {
                'season_id': season_id,
                'team_id': team_id,
                'reg_win_pct': stats['reg_win_pct'],
                'avg_point_diff': stats['avg_point_diff'],
                'fg_pct': stats['fg_pct'],
                'fg3_pct': stats['fg3_pct'],
                'ft_pct': stats['ft_pct'],
                'reb_per_game': stats['reb_per_game'],
                'ast_per_game': stats['ast_per_game'],
                'tov_per_game': stats['tov_per_game'],
                'stl_per_game': stats['stl_per_game'],
                'blk_per_game': stats['blk_per_game'],
                'fg3a_per_game': stats['fg3a_per_game'],
                'is_champion': champ_status
            }
            data.append(row)

    df_final = pd.DataFrame(data)
    
    # Convert boolean to int (True/False to 1/0) for ML compatibility
    df_final['is_champion'] = df_final['is_champion'].astype(int)
    
    return df_final

def data_cleaning_all(past_champions):
    """Clean and prepare the data for all teams with advanced stats."""
    all_teams = get_team_choices().keys()
    combined_data = []
    for team_id in all_teams:
        team_data = data_cleaning(team_id, past_champions)
        combined_data.append(team_data)
    df_final = pd.concat(combined_data, ignore_index=True)
    return df_final

def build_championship_model(df_final):
    """Build and train Random Forest model to predict championship probability."""
    
    # Prepare features (exclude non-feature columns)
    feature_columns = ['reg_win_pct', 'avg_point_diff', 'fg_pct', 'fg3_pct', 'ft_pct',
                      'reb_per_game', 'ast_per_game', 'tov_per_game', 'stl_per_game', 
                      'blk_per_game', 'fg3a_per_game']
    
    X = df_final[feature_columns]
    y = df_final['is_champion']
    
    print("Building Random Forest Championship Predictor...")
    print(f"Dataset shape: {X.shape}")
    print(f"Champions in dataset: {y.sum()}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split data (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (Random Forest doesn't require scaling, but it can help with interpretation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build Random Forest with class balancing for imbalanced dataset
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)
    
    # Model evaluation
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    print(f"Training Accuracy: {rf_model.score(X_train_scaled, y_train):.4f}")
    print(f"Testing Accuracy: {rf_model.score(X_test_scaled, y_test):.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"Test ROC AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    print(feature_importance.to_string(index=False))
    
    return rf_model, scaler, feature_columns, feature_importance

def predict_championship_probabilities(df_final, model, scaler, feature_columns, season_id=None):
    """Predict championship probabilities for all teams in a given season."""
    
    team_names = get_team_choices()
    
    if season_id is None:
        # Use the most recent season in the dataset
        season_id = df_final['season_id'].max()
    
    # Filter data for the specified season
    season_data = df_final[df_final['season_id'] == season_id].copy()
    
    if season_data.empty:
        print(f"No data found for season {season_id}")
        return None
    
    # Prepare features
    X_season = season_data[feature_columns]
    X_season_scaled = scaler.transform(X_season)
    
    # Predict probabilities
    probabilities = model.predict_proba(X_season_scaled)
    
    # Create results dataframe
    results = pd.DataFrame({
        'team_id': season_data['team_id'],
        'team_name': [team_names.get(tid, f'Team {tid}') for tid in season_data['team_id']],
        'championship_probability': probabilities[:, 1],
        'reg_win_pct': season_data['reg_win_pct'],
        'avg_point_diff': season_data['avg_point_diff'],
        'actual_champion': season_data['is_champion']
    }).sort_values('championship_probability', ascending=False)
    
    return results

def main():
    # Generate complete dataset
    print("Generating dataset...")
    df_final = data_cleaning_all(past_champions)
    df_final = df_final.sort_values('season_id', ascending=True)
    
    print(f"Dataset created with {len(df_final)} records")
    
    # Build and train the model
    model, scaler, feature_columns, feature_importance = build_championship_model(df_final)
    
    # Predict probabilities for the most recent season
    print("\n" + "="*70)
    print("CHAMPIONSHIP PROBABILITY PREDICTIONS")
    print("="*70)
    
    latest_season = df_final['season_id'].max()
    predictions = predict_championship_probabilities(df_final, model, scaler, feature_columns, latest_season)
    
    if predictions is not None:
        print(f"\nChampionship Probabilities for Season {latest_season}:")
        print("-" * 90)
        for _, row in predictions.head(10).iterrows():
            actual = "CHAMPION" if row['actual_champion'] == 1 else ""
            print(f"{row['team_name']:<25} | Prob: {row['championship_probability']:.1%} | "
                  f"Win%: {row['reg_win_pct']:.3f} | Pt Diff: {row['avg_point_diff']:+.1f} | {actual}")
    
    # You can also predict for other seasons
    print(f"\nTo predict for other seasons, call:")
    print(f"predict_championship_probabilities(df_final, model, scaler, feature_columns, season_id=22020)")
    
    return df_final, model, scaler, feature_columns

if __name__ == '__main__':
    df_final, model, scaler, feature_columns = main()