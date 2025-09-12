"""
Find all tracks that are present when the hindcast is initialised

Usage:
    find_counterfactuals.py [--basin=<str>] [--model_year=<model_year>] [--month=<month>]
    find_counterfactuals.py  (-h | --help)

Arguments:
    --basin=<str> [default: NATL]
    --model_year=<model_year>
    --month=<month>

Options:
    -h --help        Show help

"""

import datetime

from parse import parse
from tqdm import tqdm

import huracanpy
from jasmin_tracks import datasets, combine
from twinotter.util.scripting import parse_docopt_arguments

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
        str(f) for f in all_files
        if "HIND_VOR_VERTAVG_2016060900_2011060900_10" not in str(f)
        and ".old" not in str(f)
    ]

    all_tracks = []
    for fname in tqdm(all_files):
        tracks = huracanpy.load(
            fname, source="TRACK", variable_names=dataset.variable_names
        )
        tracks = combine.gather_vorticity_profile(tracks)
        tracks = tracks.hrcn.add_is_ocean().hrcn.add_basin()

        # Fix for leap years
        if "022900_" in str(fname):
            details = parse(
                str(dataset.fixed_path / leap_year_extra_path / dataset.filename),
                str(fname),
            ).named
        else:
            details = dataset.file_details(str(fname))

        # Only tracks that are initialised
        start_time = datetime.datetime(
            **{key: details[key] for key in ["year", "month", "day", "hour"]}
        )
        genesis = tracks.hrcn.get_gen_vals()
        tracks = tracks.hrcn.sel_id(genesis.track_id[genesis.time == start_time])

        # Add details to subset of tracks and save
        if len(tracks.record) > 0:
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
        else:
            print(f"Found zero initialised tracks in {fname}")

    all_tracks = huracanpy.concat_tracks(all_tracks, keep_track_id=True)

    huracanpy.save(
        all_tracks,
        f"ECMWF_HINDCASTS_initialised_tracks_"
        f"{kwargs['model_year']}_{kwargs['month']}.nc",
    )


if __name__ == "__main__":
    parse_docopt_arguments(main, __doc__)

