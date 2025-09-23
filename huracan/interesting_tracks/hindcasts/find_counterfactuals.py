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
        <unseen_fname>
        <ibtracs_fname>
        [--model_year=<model_year>]
        [--month=<month>]
    find_counterfactuals.py  (-h | --help)

Arguments:
    --model_year=<model_year>
    --month=<month>

Options:
    -h --help        Show help
"""

import datetime

import huracanpy
import numpy as np
from parse import parse
from parse_docopt import parse_docopt
from tqdm import tqdm
import xarray as xr

from jasmin_tracks import datasets, combine

from . import leap_year_extra_path


def main(unseen_fname, ibtracs_fname, **kwargs):
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

    all_tracks = huracanpy.concat_tracks(all_tracks, keep_track_id=False)

    unseen = huracanpy.load(unseen_fname)
    ibtracs = huracanpy(ibtracs_fname)

    tracks_tc, tracks_ptc, tracks_vortex = filter_tcs(all_tracks, unseen, ibtracs)

    for tracks, suffix in [
        (tracks_tc, "TC"), (tracks_ptc, "PTC"), (tracks_vortex, "vortex")
    ]:
        if len(tracks.time) > 0:
            huracanpy.save(
                tracks,
                f"ECMWF-HINDCASTS_{kwargs['model_year']}_initialised_{suffix}.nc"
            )


def filter_tcs(tracks, unseen, ibtracs):
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
    ibtracs_ptc = xr.concat(ibtracs_ptc, dim="record")

    tracks_ptc = match_initialisation(ibtracs_ptc, initial_points, tracks)

    # 3. Alternative TC
    # Grab these from the initialised "UNSEEN" tracks. They have been filtered for WCSI,
    # but not initialised vs actual unseen.
    unseen_initial = unseen.hrcn.get_gen_vals().rename(track_id="record")
    unseen_initial = unseen_initial.assign(
        track_id=("record", unseen_initial.record.values)
    )
    track_ids = unseen_initial.track_id[
        unseen_initial.forecast_start == unseen_initial.time
    ]
    unseen = unseen.hrcn.sel_id(track_ids)

    # Filter out tracks already selected
    tracks_selected = huracanpy.concat_tracks([tracks_tc, tracks_ptc], keep_track_id=False)
    matches = huracanpy.assess.match(
        [unseen, tracks_selected], ["unseen", "selected"], max_dist=0
    )

    track_ids = np.unique(unseen.track_id)
    track_ids = track_ids[
        ~np.isin(track_ids, matches.id_unseen)
    ]
    tracks_vortex = unseen.hrcn.sel_id(track_ids)

    return tracks_tc, tracks_ptc, tracks_vortex


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
    main(**parse_docopt(__doc__))

