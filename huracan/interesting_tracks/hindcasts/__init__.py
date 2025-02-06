import pathlib

# The first date in the path is when the hindcast was run and the second is the
# initialisation time. The hindcasts are run for all years back to 1995, initialised
# for the same month/year they are run, so only the year is different in the path
YYYYMMDDHH = "{year:04d}{month:02d}{day:02d}{hour:02d}"
YYYYMMDDHH_model = YYYYMMDDHH.replace("year", "model_year")
fname_pattern = (
    f"HIND_VOR_VERTAVG_{YYYYMMDDHH_model}_{YYYYMMDDHH}" + "_{ensemble_member}.nc"
)
path = pathlib.Path("../tracks/hindcasts/")
