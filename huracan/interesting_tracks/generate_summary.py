"""
Create a spreadsheet of tracks with CPS and intensification information

Usage:
    generate_summary.py <filename_in> <filename_out>
        [--npoints=<val>]
        [--basin=<str>]
        [--b_threshold=<val>] [--vtl_threshold=<val>] [--vtu_threshold=<val>]
        [--vort_threshold=<val>] [--intensification_threshold=<val>]
        [--filter_size=<val>]
        [--coherent] [--ocean]
    generate_summary.py (-h | --help)

Arguments:
    <filename>  The name of a track dataset in jasmin_tracks
    --npoints=<val>
        Number of consecutive points satisfying the selected criteria required to be
        considered a TC [default: 4]
    --basin=<str>
        One of the basins supported by huracanpy. Only consider tracks that reach
        maximum intensity in this basin, and only track points in this basin
        [default: None]
    --b_threshold=<val>
        Cyclone phase space asymmetry threshold [default: 15]
    --vtl_threshold=<val>
        Cyclone phase space low-level warm core threshold [default: 0]
    --vtu_threshold=<val>
        Cyclone phase space upper-level warm core threshold [default: 0]
    --vort_threshold=<val>
        850hPa vorticity threshold (units 1e-5) [default: 6]
    --intensification_threshold=<val>
        Rate of change of (smoothed) 850hPa vorticity with time threshold [default: 0]
    --coherent
        Require a vortex to be identified at all pressure levels [default: False]
    --ocean
        Only count track points over ocean [default: False]
    --filter_size=<val>
        Size of uniform filter to apply to cyclone phase space parameters and vorticity
        for intensification rate [default: 5]

Options:
    -h --help
        Show this screen.
"""

from itertools import groupby

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from parse_docopt import parse_docopt
import xarray as xr

import huracanpy

from huracan.cps import is_tropical_cyclone


def main(
    filename_in,
    filename_out,
    npoints=4,
    basin=None,
    b_threshold=None,
    vtl_threshold=None,
    vtu_threshold=None,
    vort_threshold=None,
    intensification_threshold=None,
    coherent=False,
    ocean=False,
    filter_size=5,
):
    tracks = huracanpy.load(filename_in)
    tc_tracks, summary = apply_filters(
        tracks,
        npoints=npoints,
        basin=basin,
        b_threshold=b_threshold,
        vtl_threshold=vtl_threshold,
        vtu_threshold=vtu_threshold,
        vort_threshold=vort_threshold,
        intensification_threshold=intensification_threshold,
        coherent=coherent,
        ocean=ocean,
        filter_size=filter_size,
    )

    summary.to_parquet(filename_out + ".parquet")
    huracanpy.save(tc_tracks, filename_out + ".nc")


def apply_filters(
    tracks,
    npoints=4,
    basin=None,
    b_threshold=None,
    vtl_threshold=None,
    vtu_threshold=None,
    vort_threshold=None,
    intensification_threshold=None,
    coherent=False,
    ocean=False,
    filter_size=5,
):
    summary = []
    tc_tracks = []

    if basin is not None and "basin" not in tracks:
        tracks = tracks.hrcn.add_basin()

    for track_id, track in tqdm(tracks.groupby("track_id")):
        if basin is not None:
            # Skip tracks that have max intensity in a different basin if filtering by
            # basin
            if track.basin[track.vorticity.argmax()] != basin:
                continue

        # Only storms that are tropical cyclones at some point in their lifecycle
        # Cyclone Phase Space
        try:
            tc = is_tropical_cyclone(
                np.abs(track.cps_b),
                track.cps_vtl,
                track.cps_vtu,
                filter_size=filter_size,
                b_threshold=b_threshold,
                vtl_threshold=vtl_threshold,
                vtu_threshold=vtu_threshold,
            )
        except (ValueError, AttributeError):
            # If no CPS thresholds are set, start with True everywhere
            tc = np.ones(len(track.time), dtype=bool)

        # Minimum vorticity
        if vort_threshold is not None:
            tc = tc & (track.relative_vorticity.sel(pressure=850) > vort_threshold)

        # Intensification rate
        if intensification_threshold is not None:
            tc = tc & (
                np.gradient(uniform_filter1d(
                    track.relative_vorticity.sel(pressure=850),
                    size=filter_size,
                    mode="nearest",
                )) > intensification_threshold
            )

        # Coherent
        if coherent:
            # Check for NaNs and mask value in TRACK (1e25)
            tc = tc & ~(
                np.isnan(track.relative_vorticity) |
                (track.relative_vorticity == 1e25)
            ).any(dim="pressure")

        # Over ocean
        if ocean:
            track.hrcn.add_is_ocean()
            tc = tc & track.is_ocean

        # Check that applied criteria are satisfied for consective npoints
        track["is_tc"] = tc
        category_consecutive = [
            (k, sum(1 for i in g)) for k, g in groupby(track.is_tc.values)
        ]

        idx = 0
        for category, count in category_consecutive:
            if category and count < npoints:
                track.is_tc[idx:idx + count] = False
            idx += count

        if basin is not None:
            is_tc = (track.is_tc & (track.basin == basin)).values.any()
        else:
            is_tc = track.is_tc.values.any()

        if is_tc:
            tc_tracks.append(track)

        times = pd.to_datetime(track.time)
        summary.append(pd.DataFrame([dict(
            track_id=track.track_id.values[0],
            storm_start=times[0],
            storm_end=times[-1],
            origin_lat=track.lat.data[0],
            origin_lon=track.lon.data[0],
            end_lat=track.lat.data[-1],
            end_lon=track.lon.data[-1],
            is_tc=is_tc,
        )]))

    summary = pd.concat(summary, ignore_index=True)
    tracks = xr.concat(tc_tracks, dim="record")

    return tracks, summary


if __name__ == "__main__":
    main(**parse_docopt(__doc__))
