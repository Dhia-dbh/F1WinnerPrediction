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