import io_fastf1
import config
from pathlib import Path
import pickle
import pandas as pd
import fastf1

fastf1.Cache.enable_cache(config.FASTF1_RAW_CACHE_DIR.as_posix())

if __name__ == "__main__":
   print("fetching sessions")
   sessions = io_fastf1.fetch_race_sessions_cache()
   print("fetched sessions:")
   for year, year_sessions in sessions.items():
      print(f"Year {year}: {len(year_sessions)} sessions")
      
   session_dump_path = config.FASTF1_CHECKPOINT_DIR / "sessions_dump.pkl"
   try:
      with open(session_dump_path, "wb") as file:
         pickle.dump(sessions, file)
   except Exception as e:
      print(f"Error saving sessions to {session_dump_path}: {e}")
