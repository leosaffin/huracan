import pathlib
import datetime

from tqdm import tqdm
from parse import parse
import numpy as np
import pandas as pd

from storm_assess import track, regions

from huracan.interesting_tracks.hindcasts import fname_pattern


def main(model_year):
    path = pathlib.Path(".")

    summary = pd.DataFrame(data=dict(
        forecast_start=[],
        storm_start=[],
        storm_end=[],
        origin_lat=[],
        origin_lon=[],
        end_lat=[],
        end_lon=[],
        max_intensity_europe=[],
    ))
    files = sorted(list(path.glob(f"*/HIND_VOR_VERTAVG_{model_year}*_*_*.nc")))
    for filepath in tqdm(files):
        details = parse(fname_pattern, filepath.name)

        start_time = datetime.datetime(
            year=details["year"],
            month=details["month"],
            day=details["day"],
            hour=details["hour"]
        )

        storms = track.load_netcdf(filepath)
        for storm in storms:
            time = pd.to_datetime(storm.time)

            over_europe = regions.landfall_europe(storm)
            min_pressure_europe = storm.mslp.data[over_europe].min()
            max_intensity_europe = storm.vmax.data[over_europe].max()

            try:
                in_tropics = np.where((storm.latitude.data < 30))
                min_pressure_tropics = storm.mslp.data[in_tropics].min()
                max_intensity_tropics = storm.vmax.data[in_tropics].max()
            except ValueError:
                # Storms with no data in tropics
                min_pressure_tropics = np.nan
                max_intensity_tropics = np.nan

            summary = pd.concat([summary, pd.DataFrame([dict(
                forecast_start=start_time,
                ensemble_member=details["ensemble_member"],
                storm_start=time[0],
                storm_end=time[-1],
                origin_lat=storm.latitude.data[0],
                origin_lon=storm.longitude.data[0],
                end_lat=storm.latitude.data[-1],
                end_lon=storm.longitude.data[-1],
                min_pressure_tropics=min_pressure_tropics,
                max_intensity_tropics=max_intensity_tropics,
                min_pressure_europe=min_pressure_europe,
                max_intensity_europe=max_intensity_europe,
            )])], ignore_index=True)

    summary.sort_values(by="forecast_start").to_csv("HIND_VOR_VERTAVG_{model_year}_summary.csv")


if __name__ == '__main__':
    main(2023)
