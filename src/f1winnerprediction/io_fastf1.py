import json
import pickle
from pathlib import Path
import logging
from datetime import datetime
import pprint

import fastf1
from fastf1.core import Session as fastf1_session
import pandas as pd
import numpy as np
import pandas.core.series
import math

import f1winnerprediction.config as config

# TODO: Implement session caching
# TODO: Refactor code and move helper function to utils.py
sessions_dump_path = config.FASTF1_CHECKPOINT_DIR / "sessions_dump.pkl"
checkpoint_dump_path = config.FASTF1_CHECKPOINT_DIR / "checkpoint.json"


def save_sessions(sessions: dict, path: Path = sessions_dump_path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "wb") as file:
		pickle.dump(sessions, file)


def load_sessions(path: Path = sessions_dump_path) -> dict[int, list[fastf1_session]]:
	if not path.exists():
		logging.warning("Sessions dump not found at %s. Returning empty cache.", path)
		return {}
	with open(path, "rb") as file:
		try:
			sessions = pickle.load(file)
			
		except EOFError as e:
			logging.error("Empty sessions dumpfile: %s", e)
			return {}
		except Exception as e:
			logging.error("Error loading sessions: %s", e)
			return {}
		
	if not _is_sessions_dict_valid(sessions):
		logging.error("Sessions dump has unexpected type: %s", type(sessions))
		return {}
	
	return sessions


def save_checkpoint(checkpoint: dict, path: Path = checkpoint_dump_path) -> None:
	# TODO: Move folder creation and path validation to another entity
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w") as file:
		json.dump(checkpoint, file)


def load_checkpoint(path: Path = checkpoint_dump_path) -> dict:
	checkpoint = {
		"year": config.YEARS_TO_FETCH[0],
		"gp_index_start": 1,
	}
	if not path.exists():
		logging.warning("Checkpoint file does not exist at %s, using defaults.", path)
		return checkpoint
	with open(path, "r") as file:
		try:
			loaded = json.load(file)
		except json.JSONDecodeError as e:
			logging.error("Invalid checkpoint JSON at %s: %s", path, e)
			return checkpoint
		except Exception as e:
			logging.error("Error loading checkpoint from %s: %s", path, e)
			return checkpoint
	if _is_checkpoint_valid(loaded):
		checkpoint = loaded
	else:
		logging.error("Checkpoint file %s does not contain a dict. Using defaults.", path)
	return checkpoint


# Fetch number of grand prix in a given year
def _extract_nb_gp_from_year(year: int) -> int:
	return fastf1.get_events_remaining(datetime(year, 1, 1))["RoundNumber"].max().item()

# Fetch number of grand prix for a list of years
def _extract_nb_gp_from_years(years: list[int]) -> list[int]:
	return [_extract_nb_gp_from_year(year) for year in years]

# Verify that checkpoint session if valid
def _is_checkpoint_valid(checkpoint) -> bool:
	return (isinstance(checkpoint, dict) and
			("year" in checkpoint and (checkpoint["year"] is None or isinstance(checkpoint["year"], int))) and
			("gp_index_start" in checkpoint and (checkpoint["gp_index_start"] is None or isinstance(checkpoint["gp_index_start"], int)))
		   )
	
# Verify that the sessions dict is valid
def _is_sessions_dict_valid(sessions: dict) -> bool:
	if not isinstance(sessions, dict):
		return False
	for year, sessions_list in sessions.items():
		if not isinstance(year, int):
			return False
		if not isinstance(sessions_list, list):
			return False
		# Allow empty sessions_list for a given year
		# if len(sessions_list) == 0:
		#     return False
		for session in sessions_list:
			if not isinstance(session, fastf1_session):
				return False
	return True

def _build_sessions_index(years: list[int], number_of_races_per_gp: list[int], checkpoint: dict[str, int] = config.DEFAULT_CHECKPOINT) -> dict[int, list[fastf1_session]]:
	sessions = {}
	count = 0
	gp_index_start = checkpoint["gp_index_start"]
	for year, nb_gp in zip(years, number_of_races_per_gp):
		sessions[year] = []
		for gp_index in range(gp_index_start, nb_gp+1):
			try:
				session = fastf1.get_session(year, gp_index, 'R')
			except Exception as e:
				logging.error(f"ERROR FETCHING SESSION: {year} {gp_index} - {e}")
				continue
			gp_name = session.event.EventName
			print(f"+------- FETCHING GP: {gp_name} {year} {gp_index}/{nb_gp} -------+")
			try:
				session.load()
			except Exception as e:
				logging.error(f"ERROR LOADING SESSION: {year} {gp_index} - {e}")
				continue
			sessions[year].append(session)
			count += 1
	
			if count % 5 == 0:
				logging.info("Saving checkpoint..")
				if _is_sessions_dict_valid(sessions):
					save_sessions(sessions)
					save_checkpoint(checkpoint)
				else:
					logging.error("Sessions dict is invalid, skipping checkpoint save.")
					pprint.pprint(sessions)
					break
				checkpoint["year"] = year
				checkpoint["gp_index_start"] = gp_index
				count = 0
	return sessions


def fetch_race_sessions_cache(years_to_fetch: list[int]) -> dict[int, list[fastf1_session]]:
	# Fetch number of grand prix per year
	number_of_races_per_gp = _extract_nb_gp_from_years(years_to_fetch)
	# Loading checkpoint if exists
	checkpoint = load_checkpoint()
	
	# Extracting progress from checkpoint
	if checkpoint["year"] is not None:
		# progression_index = checkpoint["year"] - years[0] # Assumes years are consecutive and not duplicated
		progression_index = years_to_fetch.index(checkpoint["year"])
		years_to_fetch = years_to_fetch[progression_index:]
		number_of_races_per_gp = number_of_races_per_gp[progression_index:]
		
	sessions = load_sessions()
	if not _is_sessions_dict_valid(sessions):
		logging.error("Loaded sessions cache is invalid, starting from empty cache.")
		sessions = {}
	else:
		# Loading data and updating checkpoint at each iteration
		try:
			_build_sessions_index(years_to_fetch, number_of_races_per_gp, checkpoint)
			
		except KeyboardInterrupt:
			logging.info("Fetching interrupted by user, saving checkpoint..")
			if _is_sessions_dict_valid(sessions):
				save_sessions(sessions)
				save_checkpoint(checkpoint)
	
	# Final save
	logging.info("Fetching complete, saving checkpoint..")
	if _is_sessions_dict_valid(sessions):
		save_sessions(sessions)
	save_checkpoint(checkpoint)
	return sessions

def build_drivers_dict(sessions: dict[int, list[fastf1_session]]) -> dict[str, dict]:
	# TODO: Log more information about drivers: Team, Full Name, points, nb of driven GPs.
	drivers = dict()
	for year in sessions.keys():
		count = 0
		for index, session in enumerate(sessions[year]):
			count += 1
			sessions_drivers = set(session.results["Abbreviation"])
			for session_driver in sessions_drivers:
				drivers[session_driver] = {"index": index}
	return drivers

def create_columns_windows_raceonly(df:pd.DataFrame, windows_index_start=1, windows_size=5, stride=1):
	df_windows = pd.DataFrame(columns=[f"race{i}" for i in range(windows_size)])

	# all_windows = []
	length = df.shape[1] - windows_index_start
	nb_windows = math.floor((length - windows_size)/stride) + 1
	race: pandas.core.series.Series
	for _, race in df.iterrows():
		window_index = 0
		while window_index < nb_windows:
		# for window_index in range(nb_windows):
			window_name = f"window_{window_index}"
			start_col_index = windows_index_start + window_index * stride
			end_col_index = start_col_index + windows_size
			window = race[start_col_index:end_col_index]
			# exclude windows with NaN values
			if window.isnull().any():
				# find last null within the window and set it to the current window_index
				nan_pos = np.where(window.isnull().to_numpy())[0]
				last_nan_pos = nan_pos[-1]
				print(f"Skipping window {window_name} due to NaN at position(s) {nan_pos}, jumping to window_index {window_index + last_nan_pos}")
				window_index += 1 + last_nan_pos
				continue
			df_windows.loc[len(df_windows)] = window.values  
			window_index += 1
			# print(window)	
			# print("+++++++++++++++++++")
	return df_windows