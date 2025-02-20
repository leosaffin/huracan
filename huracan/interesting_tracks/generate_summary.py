"""
Create a spreadsheet of tracks with CPS and intensification information

Usage:
    generate_summary.py <filename> <basin>
    generate_summary.py (-h | --help)

Arguments:
    <filename> The name of a track dataset in jasmin_tracks
    <basin> One of the basins supported by huracanpy

Options:
    -h --help
        Show this screen.
"""

from itertools import groupby

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from twinotter.util.scripting import parse_docopt_arguments
import xarray as xr

import huracanpy

from huracan.cps import is_tropical_cyclone


def main(filename, basin, filter_size=5):
    summary = []
    tracks = huracanpy.load(filename)
    tracks = huracanpy.trackswhere(
        tracks,
        tracks.track_id,
        lambda x: x.basin[x.vorticity.argmax()] == basin,
    )
    tc_tracks = []

    for track_id, track in tqdm(tracks.groupby("track_id")):
        is_wc_sym = False
        is_tc = False
        hits_europe = (huracanpy.info.basin(
            track.lon, track.lat, convention="Sainsbury2022MWR"
        ) == "Europe").values.any()
        vorticity = track.relative_vorticity.sel(pressure=850)

        # Only storms that are tropical cyclones at some point in their lifecycle
        tc = is_tropical_cyclone(
            np.abs(track.cps_b),
            track.cps_vtl,
            None,
            filter_size=filter_size,
            b_threshold=15,
        )
        tc = (tc & (track.basin == basin) & track.is_ocean)

        category_consecutive = [
            (k, sum(1 for i in g)) for k, g in groupby(tc.values)
        ]

        for category, count in category_consecutive:
            if category and count >= 4:
                is_wc_sym = True

        if is_wc_sym:
            vort_smoothed = uniform_filter1d(
                vorticity, size=filter_size, mode="nearest"
            )
            result = np.gradient(vort_smoothed)

            tc = tc & (result > 0)
            track["is_tc"] = tc

            category_consecutive = [
                (k, sum(1 for i in g)) for k, g in groupby(tc.values)
            ]
            idx = 0
            for category, count in category_consecutive:
                if category and count < 4:
                    track.is_tc[idx:idx + count] = False
                idx += count

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
            is_wc_sym=is_wc_sym,
            is_tc=is_tc,
            hits_europe=hits_europe,
        )]))

    summary = pd.concat(summary, ignore_index=True)
    summary.to_parquet(filename.replace(".nc", ".parquet"))

    tracks = xr.concat(tc_tracks, dim="record")
    huracanpy.save(tracks, filename.replace(".nc", "_TCs.nc"))

    track_ids = summary[summary.hits_europe == 1.0]
    tracks = tracks.isel(record=np.where(np.isin(tracks.track_id, track_ids))[0])
    huracanpy.save(tracks, filename.replace(".nc", "_TCs_hits-Europe.nc"))


if __name__ == "__main__":
    parse_docopt_arguments(main, __doc__)
