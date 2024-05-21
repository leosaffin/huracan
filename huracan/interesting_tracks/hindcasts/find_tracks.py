from itertools import groupby
import datetime

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from iris.analysis.cartography import wrap_lons
from tqdm import tqdm

from storm_assess import track, regions
from storm_assess.functions.xarray_functions import storm_in_basin
from jasmin_tracks import datasets


def add_category(storm, filter_size=None):
    storm["category"] = ("time", np.zeros(len(storm.time), dtype="<U3"))

    b = np.abs(storm.B)
    vtu = storm.TU
    vtl = storm.TL

    if filter_size is not None:
        vtu = uniform_filter1d(vtu, size=filter_size)
        vtl = uniform_filter1d(vtl, size=filter_size)
        b = uniform_filter1d(b, size=filter_size)
    # Using North Atlantic definitions from table 1 in
    # https://www.sciencedirect.com/science/article/pii/S2225603223000516
    TC = (b <= 10) & (vtl > 0) & (vtu > 0)
    ET1 = ((b > 10) & (vtl > 0))
    ET2 = ((b <= 10) & (vtl <= 0))
    EC = (b > 10) & (vtl <= 0)

    storm.category[np.where(TC)] = "TC"
    storm.category[np.where(ET1)] = "ET1"
    storm.category[np.where(ET2)] = "ET2"
    storm.category[np.where(EC)] = "EC"

    # Set weak vortices as no category. Threshold from Hodges et al. (2017)
    storm.category[np.where(storm.vorticity850hPa < 6)] = "NC"


def find_tracks():
    dataset = datasets["ECMWF_hindcasts"]
    all_files = list(dataset.find_files(model_year=2015))
    summary = pd.DataFrame(data=dict(
        model_year=[],
        forecast_start=[],
        ensemble_member=[],
        storm_id=[],
        storm_start=[],
        storm_end=[],
        origin_lat=[],
        origin_lon=[],
        end_lat=[],
        end_lon=[],
        max_intensity_europe=[],
    ))

    for fname in tqdm(all_files):
        tracks = track.load_no_assumptions(str(fname), variable_names=dataset.variable_names)
        details = dataset.file_details(str(fname))
        start_time = datetime.datetime(
            **{key: details[key] for key in ["year", "month", "day", "hour"]}
        )
        for tr in tracks:
            # Only North Atlantic storms
            if storm_in_basin(tr, "na"):
                # Only data every 12 hours for CPS in hindcasts
                times = pd.to_datetime(tr.time)
                times_ = [t for t in times if t.hour % 12 == 0]
                tr_ = tr.sel(time=times_)

                # Only storms that are tropical cyclones at some point in their lifecycle
                add_category(tr_, filter_size=3)
                category_consecutive = [(k, sum(1 for i in g)) for k, g in groupby(tr_.category.data)]
                tcident = False
                for category, count in category_consecutive:
                    if category == "TC" and count > 1:
                        tcident = True
                if tcident:
                    in_tropics = tr.latitude < 30
                    if in_tropics.any():
                        min_pressure_tropics = tr.mslp[in_tropics].data.min()
                        max_intensity_tropics = tr.vmax10m[in_tropics].data.max()
                    else:
                        min_pressure_tropics = np.nan
                        max_intensity_tropics = np.nan

                    # 36–70 deg N and 10 deg W–30 deg E
                    lons = wrap_lons(tr.longitude, -180, 360)
                    in_europe = (-10 < lons) & (lons < 30) & (36 < tr.latitude) & (tr.latitude < 70)
                    if in_europe.any():
                        min_pressure_europe = tr.mslp[in_europe].data.min()
                        max_intensity_europe = tr.vmax10m[in_europe].data.max()
                    else:
                        min_pressure_europe = np.nan
                        max_intensity_europe = np.nan
                    summary = pd.concat([summary, pd.DataFrame([dict(
                        model_year=details["model_year"],
                        forecast_start=start_time,
                        ensemble_member=details["ensemble_member"],
                        storm_start=times[0],
                        storm_end=times[-1],
                        origin_lat=tr.latitude.data[0],
                        origin_lon=tr.longitude.data[0],
                        end_lat=tr.latitude.data[-1],
                        end_lon=tr.longitude.data[-1],
                        min_pressure_tropics=min_pressure_tropics,
                        max_intensity_tropics=max_intensity_tropics,
                        min_pressure_europe=min_pressure_europe,
                        max_intensity_europe=max_intensity_europe,
                    )])], ignore_index=True)

    summary.to_csv("interesting_tracks_ECMWF_Hindcasts_2015.csv")


if __name__ == '__main__':
    find_tracks()
