import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_label_split(df_windows: pd.DataFrame):
	X = df_windows.iloc[:, :-1].values
	y = df_windows.iloc[:, -1].values
	return X, y

def normalize_features(X: pd.DataFrame):
	scaler = StandardScaler()
	X_normalized = scaler.fit_transform(X)
	return X_normalized

def create_race_time_windows(sessions: dict, map_gp_indices: dict[str, dict[int, int]], window_size: int = 2) -> pd.DataFrame:
	"""
	Create time windows of lap times for each driver
	"""
	session_last_year = sessions[2023][0]  # First session of 2023
	mean_lap_time_last_year = session_last_year.laps["LapTime"].dt.total_seconds().groupby(session_last_year.laps["Driver"]).mean().sort_values()
	df_lap_time_last_year = mean_lap_time_last_year.reset_index()
	session_current_year = sessions[2024][0]  # First session of 2024
	mean_lap_time_current_year = session_current_year.laps["LapTime"].dt.total_seconds().groupby(session_current_year.laps["Driver"]).mean().sort_values()
	df_lap_time_current_year = mean_lap_time_current_year.reset_index()
	data = df_lap_time_last_year.merge(df_lap_time_current_year, on="Driver", suffixes=('_2023', '_2024'))