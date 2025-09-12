"""
Separate initialised tracks into three categories

1. Initialised as TC. Matches ERA5 while classified as a tropical storm in IBTrACS
2. Initialised PTC. Matches ERA5 after the IBTrACS track is no longer classified as a
   tropical storm
3. Alternative TC. Tracks that are not initialised as a TC, but develop into TCs in the
   forecast (warm core, symmetric, intensifying). This will include tracks that match
   the early stages of real TC, invests, and vortices that did not develop in reality

Usage:
    find_counterfactuals.py <initialised_fname> <unseen_fname> <eibtracs_fname>
    find_counterfactuals.py  (-h | --help)

Arguments:
    <initialised_fname>
    <unseen_fname>
    <eibtracs_fname>

Options:
    -h --help        Show help
"""

import pathlib

import huracanpy
import numpy as np
from parse_docopt import parse_docopt
from tqdm import tqdm
import xarray as xr


def main(initialised_fname, unseen_fname, eibtracs_fname):
    tracks = huracanpy.load(initialised_fname)

    # Make sure track_id is not a coordinate or the matching fails
    initial_points = tracks.hrcn.get_gen_vals().rename(track_id="record")
    initial_points = initial_points.assign(
        track_id=("record", initial_points.record.values)
    )

    eibtracs = huracanpy.load(eibtracs_fname).sel(dataset="TRACK-ERA5")

    # Drop NaNs
    eibtracs = eibtracs.isel(
        record=np.where(~np.isnan(eibtracs.lon))[0]
    )

    # 1. Initialised as a TC
    eibtracs_tc = eibtracs.isel(
        record=np.where(eibtracs.nature == "TS")[0]
    )

    tracks_tc = match_initialisation(eibtracs_tc, initial_points, tracks)

    # Account for two tracks initialised close together
    # Only select the closest point for a given hindcast

    # 2. Initialised post TC
    # Get extended IBTrACS after the last tropical storm tag
    # Only use tracks with at least one tropical storm tag
    eibtracs = eibtracs.hrcn.sel_id(
        np.unique(eibtracs_tc.track_id)
    )

    eibtracs_ptc = []
    for track_id, track in eibtracs.groupby("track_id"):
        idx = np.where(track.nature == "TS")[0][-1]
        if idx < len(track.time) - 1:
            eibtracs_ptc.append(track.isel(record=slice(idx + 1, None)))
    eibtracs_ptc = xr.concat(eibtracs_ptc, dim="record")

    tracks_ptc = match_initialisation(eibtracs_ptc, initial_points, tracks)

    for tracks, suffix in [
        (tracks_tc, "TC"), (tracks_ptc, "PTC"),
    ]:
        if len(tracks.time) > 0:
            huracanpy.save(tracks, initialised_fname.replace(".nc", f"_initialised_{suffix}.nc"))


def match_initialisation(ibtracs, initial_points, hindcast_tracks):
    matches = huracanpy.assess.match(
        [ibtracs, initial_points], ["ibtracs", "hindcast"]
    )
    tracks = hindcast_tracks.hrcn.sel_id(matches.id_hindcast)

    # Add IBTrACS info
    tracks["ibtracs_id"] = (
        "record", np.zeros(len(tracks.time), dtype=ibtracs.track_id.dtype)
    )

    for n, row in matches.iterrows():
        tracks.ibtracs_id[tracks.track_id == row.id_hindcast] = row.id_ibtracs

    return tracks


def merge_subsets():
    for year in tqdm(range(2015, 2024 + 1)):
        for suffix in ["TC", "PTC"]:
            files = [
                str(f) for f in pathlib.Path(".").glob(
                    f"ECMWF_HINDCASTS_initialised_tracks_"
                    f"{year}_*_initialised_{suffix}.nc"
                )
            ]
            tracks = huracanpy.load(files)
            huracanpy.save(tracks, f"ECMWF-HINDCASTS_{year}_initialised_{suffix}.nc")


def unseen_initialised(initialised_fname, unseen_fname, eibtracs_fname):
    other_tracks = huracanpy.load(
        [initialised_fname + "TC.nc", initialised_fname + "PTC.nc"]
    )

    # 3. Alternative TC
    # Grab these from the initialised "UNSEEN" tracks. They have been filtered for WCSI,
    # but not initialised vs actual unseen.
    unseen_tracks = huracanpy.load(unseen_fname)
    unseen_tracks_initial = unseen_tracks.hrcn.get_gen_vals().rename(track_id="record")
    unseen_tracks_initial = unseen_tracks_initial.assign(
        track_id=("record", unseen_tracks_initial.record.values)
    )
    track_ids = unseen_tracks_initial.track_id[
        unseen_tracks_initial.forecast_start == unseen_tracks_initial.time
    ]
    unseen_tracks = unseen_tracks.hrcn.sel_id(track_ids)

    # Filter out tracks already selected
    matches = huracanpy.assess.match(
        [unseen_tracks, other_tracks], ["unseen", "ibtracs"], max_dist=0
    )

    track_ids = np.unique(unseen_tracks.track_id)
    track_ids = track_ids[
        ~np.isin(track_ids, matches.id_ibtracs)
    ]
    tracks_vortex = unseen_tracks.hrcn.sel_id(track_ids)
    huracanpy.save(tracks_vortex, initialised_fname + "vortex.nc")


if __name__ == "__main__":
    #main(**parse_docopt(__doc__))
    #merge_subsets()
    unseen_initialised(**parse_docopt(__doc__))
