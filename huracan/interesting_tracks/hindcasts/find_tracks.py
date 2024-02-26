import pathlib

from storm_assess import track
from storm_assess.regions import landfall_europe
from storm_assess.functions.xarray_functions import storm_in_basin

parent_path = pathlib.Path("/gws/nopw/j04/huracan/data/tracks/tropical_cyclones/TRACK/ECMWF-HINDCASTS/TC/")
filename = "tr_trs_pos.2day_addwinds_addmslp.highres.new"

variable_names = [
    "vmax",
    "v10m",
    "mslp",
]


def find_tracks():
    for model_year in range(2015, 2023+1):
        for run_date_path in parent_path.glob(f"{model_year}*"):
            output_path = pathlib.Path("../data/tracks/") / run_date_path.name
            if output_path.exists():
                print(f"{run_date_path} completed")
            else:
                output_path.mkdir()
                for start_date_path in run_date_path.glob("*" + run_date_path.name[4:]):
                    print(start_date_path)
                    for ensemble_path in start_date_path.glob(f"HIND_VOR_VERTAVG_{run_date_path.name}_{start_date_path.name}_*"):
                        try:
                            tr = track.load_no_assumptions(str(ensemble_path / filename), variable_names=variable_names)
                        except FileNotFoundError as e:
                            try:
                                tr = track.load_no_assumptions(str(ensemble_path / filename) + ".gz", variable_names=variable_names)
                            except FileNotFoundError:
                                print(e)
                                continue

                        # North Atlantic Storms that affect Europe
                        interesting_tracks = [t for t in tr if storm_in_basin(t, "na") and landfall_europe(t).any()]
                        if len(interesting_tracks) > 0:
                            print(f"{len(interesting_tracks)} tracks in {ensemble_path.name}")
                            track.save_netcdf(interesting_tracks, output_path / f"{ensemble_path.name}.nc")
                        else:
                            print(f"No matching tracks in {ensemble_path.name}")


if __name__ == '__main__':
    find_tracks()
