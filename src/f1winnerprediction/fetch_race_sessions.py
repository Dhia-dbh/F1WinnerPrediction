import io_fastf1
import config
from pathlib import Path
import pickle
import pandas as pd
import fastf1
import argparse

fastf1.Cache.enable_cache(config.FASTF1_RAW_CACHE_DIR.as_posix())
def fetch_race_sessions_per_year(year: int) -> None:
   print("fetching sessions")
   sessions = io_fastf1.fetch_race_sessions_cache([year], use_checkpoint=False, use_sessions_cache=False)
   if not io_fastf1._is_sessions_dict_valid(sessions) or len(sessions.keys()) == 0:
      raise ValueError("Fetched sessions dictionary is not valid.")
   print("fetched sessions:")
   for year, year_sessions in sessions.items():
      print(f"Year {year}: {len(year_sessions)} sessions")
   
   session_dump_name = f"sessions_dump_{year}.pkl"
   session_dump_path = config.FASTF1_CHECKPOINT_DIR / session_dump_name
   try:
      with open(session_dump_path, "wb") as file:
         pickle.dump(sessions, file)
   except Exception as e:
      print(f"Error saving sessions to {session_dump_path}: {e}")


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--year", type=int, help="Year to fetch (optional)")
   args = parser.parse_args()
   fetch_race_sessions_per_year(args.year)