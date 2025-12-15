from requests_cache import logger


def map_gp_indices_between_years(sessions: dict, year_start: int | None = None, year_end: int | None = None) -> dict[str, dict[int, int]]:
	"""
	Maps a given year to a Grand Prix index.

	Args:
		year_start (int): The first year.
		year_end (int): The second year.

	Returns:
		dict[int, int]: A dictionary mapping 2 consecutive years to their Grand Prix index.
	"""
	years = list(sessions.keys())
	if year_start is None:
		year_start = min(years)
	if year_end is None:
		year_end = max(years) + 1
  
   # assert type(year_start) == int, "year_start must be integer"
   # assert type(year_end) == int, "year_end must be integer" 
   
	gp_index_map = {}
	year_index = year_start # year counter
	while year_index < year_end:
		if year_index not in years or (year_index + 1) not in years:
			year_index += 1
			continue
		for index_session_year_1, session_year_1 in enumerate(sessions[year_index]):
			for index_session_year_2, session_year_2 in enumerate(sessions[year_index + 1]):
				session_year_1_name = 	session_year_1.event['EventName']
				session_year_2_name = session_year_2.event['EventName']
				if session_year_1_name == session_year_2_name:
					# print(f"Mapping GP '{session_year_1_name}' between {year_index} and {year_index + 1}: {index_session_year_1} -> {index_session_year_2}")
					if f"{year_index}-{year_index + 1}" in gp_index_map:
						gp_index_map[f"{year_index}-{year_index + 1}"][index_session_year_1] = index_session_year_2
					else:
						gp_index_map[f"{year_index}-{year_index + 1}"] = {
							index_session_year_1: index_session_year_2
						}
		if len(gp_index_map[f"{year_index}-{year_index + 1}"]) < max(len(sessions[year_index]), len(sessions[year_index])):
			logger.warning(f"GP index mapping incomplete for years {year_index} and {year_index + 1}. \
      Some GPs may be missing.")
		year_index += 1
	return gp_index_map
	 
		