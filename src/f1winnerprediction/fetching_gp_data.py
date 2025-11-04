# import torch
import numpy as np
import pandas as pd
import fastf1
import fastf1.core
from typing import List
from datetime import datetime
from pathlib import Path
import pickle
import config
import logging

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

fastf1.Cache.enable_cache(config.CACHE_DIR)

YEARS : List = [2021, 2022, 2023, 2024, 2025]
# YEARS : List = [2021]
NUMBER_OF_GP = [fastf1.get_events_remaining(datetime(year, 1, 1))["RoundNumber"].max().item() for year in YEARS]

checkpoint = {
	"year": 2021,
	"gp_index_start": 1
}
sessions: dict[int, list[fastf1.core.Session]] = {}
sessions_dump_path = config.CHECKPOINT_DIR / "sessions_dump.pkl"


def save_checkpoint(checkpoint: dict = sessions, path: Path = sessions_dump_path):
	with open(path, "wb") as file:
	  pickle.dump(checkpoint, file)
	  
def load_checkpoint(path: Path = sessions_dump_path) -> dict:
	checkpoint = {}
	with open(path, "rb") as file:
		try:
			checkpoint = pickle.load(file)
		except EOFError as e:
			logging.error(f"Empty dumpfile: {e}")
		except Exception as e:
			logging.error(f"Error loading checkpoint: {e}")

	return checkpoint

def fetch_gp_data(checkpoint: dict = checkpoint) -> fastf1.core.Session:
	# Load All needed Data
	sessions = {}
	years = YEARS
	number_of_gp = NUMBER_OF_GP
	gp_index_start = 1

	# Global trackers for year and gp index in case of interruption
	global_gp_index = gp_index_start
	global_year = 2021

	# Loading checkpoint
	# If checkpoint exists, update parameters
	if checkpoint["year"] is not None:
		progression_index = YEARS.index(checkpoint["year"])
		years = YEARS[progression_index:]
		number_of_gp = NUMBER_OF_GP[progression_index:]
  
	if checkpoint["gp_index_start"] is not None:
		gp_index_start = checkpoint["gp_index_start"]
		
	# Loading data and updating checkpoint at each iteration
	try:
		count = 0
		for year, nb_gp in zip(years, number_of_gp):
			global_year = year
			sessions[year] = []
			for gp_index in range(gp_index_start, nb_gp+1):
				global_gp_index = gp_index
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
					save_checkpoint(sessions, sessions_dump_path)
					checkpoint["year"] = year
					checkpoint["gp_index_start"] = gp_index
					count = 0
	except KeyboardInterrupt as e:
		logging.info("Fetching interrupted by user, saving checkpoint..")
		save_checkpoint(sessions, sessions_dump_path)
		checkpoint = {
			"year": global_year,
			"gp_index_start": global_gp_index
		}
		print(f"Checkpoint saved at year: {global_year}, gp_index: {global_gp_index}")
	# Final save
	logging.info("Fetching complete, saving checkpoint..")
	save_checkpoint(sessions, sessions_dump_path)
	return sessions

def main():
	# GP sessions
	sessions: dict[int, list[fastf1.core.Session]] = {}
	Path.mkdir(config.CHECKPOINT_DIR, exist_ok=True)
	sessions_dump_path = config.CHECKPOINT_DIR / "sessions_dump.pkl"
	try:
		sessions_dump_path.touch(exist_ok=False)
		is_dump_present = False
	except FileExistsError:
		is_dump_present = True
		
	# Verify that the dump is complete
	if is_dump_present:
		load_checkpoint(sessions_dump_path)
		#Make sure all years and gp are present
		# assert sessions.keys() == YEARS, "Missing years in the dump"
		if sessions.keys() == YEARS:
			logging.warning("Missing years in the dump")
		# assert all(len(sessions[year]) == NUMBER_OF_GP[i] for i, year in enumerate(YEARS)), "Missing GPs in the dump"
		if sessions.keys() != YEARS or not all(len(sessions[year]) == NUMBER_OF_GP[i] for i, year in enumerate(YEARS)):
			logging.warning("Missing GPs in the dump")
	sessions = fetch_gp_data()
   

if __name__ == "__main__":
	main()
