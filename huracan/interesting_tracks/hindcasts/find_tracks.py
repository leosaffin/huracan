"""

Usage:
    find_tracks.py [--basin=<str>] [--model_year=<model_year>] [--year=<year>] [--month=<month>] [--day=<day>]
    find_tracks.py  (-h | --help)

Arguments:
    --basin=<str> [default: NATL]
    --model_year=<model_year>
    --year=<year>
    --month=<month>
    --day=<day>

Options:
    -h --help        Show help

"""

import datetime

from parse import parse
import numpy as np
import xarray as xr
from tqdm import tqdm

import huracanpy
from jasmin_tracks import datasets
from twinotter.util.scripting import parse_docopt_arguments

from huracan.interesting_tracks.find_tracks import tidy_track_metadata
from huracan.interesting_tracks.generate_summary import apply_filters

_YYYYMMDDHH = "{year:04d}{month:02d}{day:02d}{hour:02d}"
_YYYYMMDDHH_model = _YYYYMMDDHH.replace("year", "model_year").replace(
    "day", "model_day"
)

leap_year_extra_path = (
    f"{_YYYYMMDDHH_model}/{_YYYYMMDDHH}/"
    f"HIND_VOR_VERTAVG_{_YYYYMMDDHH_model}_{_YYYYMMDDHH}" + "_{ensemble_member}/"
)


def main(basin="NATL", **kwargs):
    keys = kwargs.keys()
    for key in keys:
        kwargs[key] = int(kwargs[key])

    dataset = datasets["ECMWF_hindcasts"]
    plevs = [
        float(result.named["n"]) for result in
        [parse("vorticity{n}hPa", var) for var in dataset.variable_names]
        if result is not None
    ]
    all_files = list(dataset.find_files(**kwargs))
    all_files = [
        f for f in all_files
        if "HIND_VOR_VERTAVG_2016060900_2011060900_10" not in str(f)
        and ".old" not in str(f)
    ]

    all_tracks = []
    current_track_id = 1
    for fname in tqdm(all_files):
        tracks = huracanpy.load(
            str(fname), source="TRACK", variable_names=dataset.variable_names
        )
        tracks = tidy_track_metadata(tracks, plevs)
        tracks = tracks.hrcn.add_is_ocean().hrcn.add_basin()

        # Fix for leap years
        if "022900_" in str(fname):
            details = parse(
                str(dataset.fixed_path / leap_year_extra_path / dataset.filename),
                str(fname),
            ).named
        else:
            details = dataset.file_details(str(fname))

        tracks_12hr = tracks.isel(record=np.where(tracks.time.dt.hour % 12 == 0)[0])
        try:
            tracks_tc, summary = apply_filters(
                tracks_12hr,
                npoints=3,
                basin=basin,
                b_threshold=15,
                vtl_threshold=0,
                vtu_threshold=0,
                vort_threshold=6,
                intensification_threshold=0,
                coherent=True,
                ocean=False,
                filter_size=3,
            )

            # Add the "is_TC" info back to the full tracks
            tracks = tracks.isel(record=np.where(
                np.isin(tracks.track_id, summary[summary.is_tc == 1].track_id)
            )[0])
            tracks["is_tc"] = ("record", np.zeros(len(tracks.time), dtype=bool))
            tracks.is_tc[tracks.time.dt.hour % 12 == 0] = tracks_tc.is_tc.values

            # Reindex track_ids
            track_ids, new_track_ids = np.unique(tracks.track_id, return_inverse=True)
            tracks["track_id_original"] = ("record", tracks.track_id.values)
            tracks["track_id"] = ("record", new_track_ids + current_track_id)
            tracks.track_id.attrs["cf_role"] = "trajectory_id"

            current_track_id = tracks.track_id.values.max() + 1

            # Add details to subset of tracks and save
            start_time = datetime.datetime(
                **{key: details[key] for key in ["year", "month", "day", "hour"]}
            )
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

            all_tracks.append(tracks)
        except ValueError as e:
            print(e)
            print(f"No interesting tracks found for", details, "\n\n")

    if len(all_tracks) == 0:
        print("No interesting tracks found")

    else:
        all_tracks = xr.concat(all_tracks, dim="record")

        huracanpy.save(
            all_tracks,
            f"hindcast_tracks_{basin}_TC_{kwargs['model_year']}_{kwargs['month']}.nc",
        )


if __name__ == "__main__":
    parse_docopt_arguments(main, __doc__)
