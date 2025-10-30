import pandas as pd
from pathlib import Path

def write_to_file(output: str):
	if type(output) == pd.DataFrame:
		output.to_csv("output.csv")
		return
	if type(output) == str:
		with open("output.txt", "w") as f:
			f.write(output)
			return
	print(f"UNSUPPORTED OBJECT TYPE:  {type(output)}")


