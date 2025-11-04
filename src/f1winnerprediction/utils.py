import pandas as pd
from pathlib import Path

def write_to_file(output: pd.DataFrame | str):
	if isinstance(output, pd.DataFrame):
		output.to_csv("output.csv")
		return
	if isinstance(output, str):
		with open("output.txt", "w") as f:
			f.write(output)
			return
	# TODO: THROW EXCEPTION
	print(f"UNSUPPORTED OBJECT TYPE:  {type(output)}")


