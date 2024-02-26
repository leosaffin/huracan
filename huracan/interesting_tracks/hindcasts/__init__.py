import pathlib

from storm_assess import track

# The first date in the path is when the hindcast was run and the second is the
# initialisation time. The hindcasts are run for all years back to 1995, initialised
# for the same month/year they are run, so only the year is different in the path
YYYYMMDDHH = "{year:04d}{month:02d}{day:02d}{hour:02d}"
YYYYMMDDHH_model = YYYYMMDDHH.replace("year", "model_year")
fname_pattern = f"HIND_VOR_VERTAVG_{YYYYMMDDHH_model}_{YYYYMMDDHH}" + "_{ensemble_member}.nc"
path = pathlib.Path("../tracks/hindcasts/")


def load_from_row(row, model_year=2015):
    storms = track.load_netcdf(path / (YYYYMMDDHH_model + "/" + fname_pattern).format(
        model_year=model_year,
        year=row.forecast_start.year,
        month=row.forecast_start.month,
        day=row.forecast_start.day,
        hour=row.forecast_start.hour,
        ensemble_member=row.ensemble_member
    ))

    for storm in storms:
        if storm.longitude.data[0] == row.origin_lon:
            return storm
