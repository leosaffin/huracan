"""

Usage:
    find_tracks.py [--model_year=<model_year>] [--year=<year>] [--month=<month>] [--day=<day>]
    find_tracks.py  (-h | --help)

Arguments:
    --model_year=<model_year>
    --year=<year>
    --month=<month>
    --day=<day>

Options:
    -h --help        Show help

"""

import datetime
from itertools import groupby

from parse import parse
import numpy as np
from scipy.ndimage import uniform_filter1d
import xarray as xr
from tqdm import tqdm

import huracanpy
from jasmin_tracks import datasets
from twinotter.util.scripting import parse_docopt_arguments

from huracan.cps import is_tropical_cyclone
from huracan.interesting_tracks.find_tracks import tidy_track_metadata

_YYYYMMDDHH = "{year:04d}{month:02d}{day:02d}{hour:02d}"
_YYYYMMDDHH_model = _YYYYMMDDHH.replace("year", "model_year").replace(
    "day", "model_day"
)

leap_year_extra_path = (
    f"{_YYYYMMDDHH_model}/{_YYYYMMDDHH}/"
    f"HIND_VOR_VERTAVG_{_YYYYMMDDHH_model}_{_YYYYMMDDHH}" + "_{ensemble_member}/"
)


def main(**kwargs):
    keys = kwargs.keys()
    for key in keys:
        kwargs[key] = int(kwargs[key])

    dataset = datasets["ECMWF_hindcasts"]
    all_files = list(dataset.find_files(**kwargs))
    all_files = [
        f for f in all_files
        if "HIND_VOR_VERTAVG_2016060900_2011060900_10" not in str(f)
        and ".old" not in str(f)
    ]

    all_tracks = None
    current_track_id = 1
    for fname in tqdm(all_files):
        tracks = huracanpy.load(
            str(fname), source="TRACK", variable_names=dataset.variable_names
        )
        tracks.hrcn.add_is_ocean()
        tracks.hrcn.add_basin()

        # Fix for leap years
        if "022900_" in str(fname):
            details = parse(
                str(dataset.fixed_path / leap_year_extra_path / dataset.filename),
                str(fname),
            ).named
        else:
            details = dataset.file_details(str(fname))

        start_time = datetime.datetime(
            **{key: details[key] for key in ["year", "month", "day", "hour"]}
        )

        # Filter tracks
        condition = tracks.is_ocean & (tracks.basin == "NATL")
        track_ids = np.unique(tracks.isel(record=np.where(condition)[0]).track_id)
        condition = np.isin(tracks.track_id, track_ids)
        tracks = tracks.isel(record=np.where(condition)[0])

        track_ids = []
        tracks_12hr = tracks.isel(record=np.where(tracks.time.dt.hour % 12 == 0)[0])
        for track_id, track in tracks_12hr.groupby("track_id"):
            # Only storms that are tropical cyclones at some point in their lifecycle
            vort_smoothed = uniform_filter1d(
                track.vorticity850hPa, size=3, mode="nearest"
            )

            condition = (
                is_tropical_cyclone(
                    np.abs(track.cps_b),
                    track.cps_vtl,
                    None,
                    filter_size=3,
                    b_threshold=15
                )
                & track.is_ocean
                & (track.basin == "NATL")
                & (np.gradient(vort_smoothed) > 0)
            )

            category_consecutive = [
                (k, sum(1 for i in g)) for k, g in groupby(condition.values)
            ]

            is_tc = False
            for category, count in category_consecutive:
                # Discount is_tc if less than 4 timesteps
                if category and count > 3:
                    is_tc = True

            if is_tc:
                track_ids.append(track_id)

        if len(track_ids) > 0:
            tracks = tracks.isel(record=np.where(np.isin(tracks.track_id, track_ids))[0])

            # Reindex track_ids
            tracks = tracks.sortby("track_id")
            track_ids, new_track_ids = np.unique(tracks.track_id, return_inverse=True)
            tracks["track_id_original"] = tracks.track_id
            del tracks.track_id_original.attrs["cf_role"]
            tracks["track_id"] = ("record", new_track_ids + current_track_id)
            tracks.track_id.attrs["cf_role"] = "trajectory_id"

            current_track_id = tracks.track_id.values.max() + 1

            # Add details to subset of tracks and save
            if details["ensemble_member"] == "CNTRL":
                details["ensemble_member"] = "0"

            tracks["forecast_start"] = ("record", [start_time] * len(tracks.record))
            tracks["model_year"] = (
                "record",
                [int(details["model_year"])] * len(tracks.record),
            )
            tracks["ensemble_member"] = (
                "record",
                [int(details["ensemble_member"])] * len(tracks.record),
            )

            if all_tracks is None:
                all_tracks = tracks
            else:
                all_tracks = xr.concat([all_tracks, tracks], dim="record")

    if all_tracks is None:
        print("No interesting tracks found")

    else:
        plevs = [
            result.named["n"] for result in
            [parse("vorticity{n}hPa", var) for var in dataset.variable_names]
            if result is not None
        ]
        tracks = tidy_track_metadata(all_tracks, plevs)

        huracanpy.save(
            tracks,
            f"hindcast_tracks_NATL_TC_{kwargs['model_year']}_{kwargs['month']}.nc",
        )


if __name__ == "__main__":
    parse_docopt_arguments(main, __doc__)
