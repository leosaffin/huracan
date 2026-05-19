"""
Find all tracks that are present when the hindcast is initialised and separate into
three categories

1. Initialised as TC. Matches ERA5 while classified as a tropical storm in IBTrACS
2. Initialised PTC. Matches ERA5 after the IBTrACS track is no longer classified as a
   tropical storm
3. Alternative TC. Tracks that are not initialised as a TC, but develop into TCs in the
   forecast (warm core, symmetric, intensifying). This will include tracks that match
   the early stages of real TC, invests, and vortices that did not develop in reality

Usage:
    find_counterfactuals.py
        <ibtracs_fname>
        [--model_year=<model_year>]
        [--month=<month>]
    find_counterfactuals.py  (-h | --help)

Arguments:
    <ibtracs_fname>
    --model_year=<model_year>
    --month=<month>

Options:
    -h --help        Show help
"""

import datetime

import huracanpy
import numpy as np
import pandas as pd
from parse import parse
from parse_docopt import parse_docopt
from tqdm import tqdm
import xarray as xr

from jasmin_tracks import datasets, combine

from . import leap_year_extra_path


def main(ibtracs_fname, **kwargs):
    dataset = datasets["ECMWF_hindcasts"]
    all_files = list(dataset.find_files(**kwargs))
    all_files = [
        str(f) for f in all_files
        if "HIND_VOR_VERTAVG_2016060900_2011060900_10" not in str(f)
        and ".old" not in str(f)
    ]

    ibtracs = huracanpy.load(ibtracs_fname)

    all_tracks_tc = []
    all_tracks_ptc = []
    for fname in tqdm(all_files):
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

        ibtracs_ = ibtracs.track_id[pd.to_datetime(ibtracs.time) == start_time]
        if len(ibtracs_.record) > 0:
            ibtracs_ = ibtracs.hrcn.sel_id(ibtracs_)
            tracks = huracanpy.load(
                fname, source="TRACK", variable_names=dataset.variable_names
            )
            tracks = combine.gather_vorticity_profile(tracks)
            tracks = tracks.hrcn.add_is_ocean().hrcn.add_basin()

            # Only tracks that are initialised
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

                tracks_tc, tracks_ptc = filter_tcs(tracks, ibtracs_)

                if len(tracks_tc.time) > 0:
                    all_tracks_tc.append(tracks_tc)
                if len(tracks_ptc.time) > 0:
                    all_tracks_ptc.append(tracks_ptc)
            else:
                print(f"Found zero initialised tracks in {fname}")
        else:
            print(f"No active tracks for {fname}")

    for tracks, suffix in [
        (all_tracks_tc, "TC"), (all_tracks_ptc, "PTC")
    ]:
        if len(tracks) > 0:
            tracks = huracanpy.concat_tracks(tracks, keep_track_id=False)
            huracanpy.save(
                tracks,
                f"ECMWF-HINDCASTS_{kwargs['model_year']}-{kwargs['month']}_initialised_{suffix}.nc"
            )


def filter_tcs(tracks, ibtracs):
    # Make sure track_id is not a coordinate or the matching fails
    initial_points = tracks.hrcn.get_gen_vals().rename(track_id="record")
    initial_points = initial_points.assign(
        track_id=("record", initial_points.record.values)
    )

    # 1. Initialised as a TC
    ibtracs_tc = ibtracs.isel(record=np.where(ibtracs.nature == "TS")[0])
    tracks_tc = match_initialisation(ibtracs_tc, initial_points, tracks)

    # 2. Initialised post TC
    # Get IBTrACS after the last tropical storm tag
    ibtracs_ptc = []
    for track_id, track in ibtracs.groupby("track_id"):
        idx = np.where(track.nature == "TS")[0][-1]
        if idx < len(track.time) - 1:
            ibtracs_ptc.append(track.isel(record=slice(idx + 1, None)))

    if len(ibtracs_ptc) > 0:
        ibtracs_ptc = xr.concat(ibtracs_ptc, dim="record")
        tracks_ptc = match_initialisation(ibtracs_ptc, initial_points, tracks)
    else:
        tracks_ptc = ibtracs.isel(record=slice(0, 0))

    return tracks_tc, tracks_ptc


def match_initialisation(ibtracs, initial_points, hindcast_tracks):
    matches = huracanpy.assess.match(
        [ibtracs, initial_points], ["ibtracs", "hindcast"], max_dist=165
    )
    tracks = hindcast_tracks.hrcn.sel_id(matches.id_hindcast)

    # Add IBTrACS info
    tracks["ibtracs_id"] = (
        "record", np.zeros(len(tracks.time), dtype=ibtracs.track_id.dtype)
    )

    for n, row in matches.iterrows():
        tracks.ibtracs_id[tracks.track_id == row.id_hindcast] = row.id_ibtracs

    return tracks


if __name__ == "__main__":
    kwargs = parse_docopt(__doc__)
    print(kwargs)
    main(**kwargs)


