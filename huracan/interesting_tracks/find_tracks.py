"""
First sweep to filter interesting tracks

Subset of tracks that have a point over ocean in the specified basin

Usage:
    find_tracks.py <dataset_name> <basin>
    find_tracks.py (-h | --help)

Arguments:
    <dataset_name> The name of a track dataset in jasmin_tracks
    <basin> One of the basins supported by huracanpy

Options:
    -h --help
        Show this screen.
"""

import warnings

from parse import parse
import numpy as np
import xarray as xr
from tqdm import tqdm
from twinotter.util.scripting import parse_docopt_arguments

import huracanpy
from jasmin_tracks import datasets


def main(dataset_name, basin):
    dataset = datasets[dataset_name]
    all_files = list(dataset.find_files())

    tracks = filter_by_basin(
        all_files, basin, source="TRACK", variable_names=dataset.variable_names
    )

    plevs = [
        result.named["n"] for result in
        [parse("vorticity{n}hPa", var) for var in dataset.variable_names]
        if result is not None
    ]
    tracks = tidy_track_metadata(tracks, plevs)

    huracanpy.save(tracks, f"{dataset_name}_tracks_NATL.nc")


def filter_by_basin(files, basin, **kwargs):
    current_track_id = 1
    for n, fname in tqdm(enumerate(files)):
        tracks = huracanpy.load(str(fname), **kwargs)

        # Add required information
        tracks = tracks.hrcn.add_is_ocean()
        tracks = tracks.hrcn.add_basin()

        # Filter tracks
        # Must contain at least one point over ocean in the specified basin
        condition = tracks.is_ocean & (tracks.basin == basin)
        track_ids = np.unique(tracks.isel(record=np.where(condition)[0]).track_id)
        condition = np.isin(tracks.track_id, track_ids)
        tracks = tracks.isel(record=np.where(condition)[0])

        # Reindex track_ids
        tracks = tracks.sortby("track_id")
        track_ids, new_track_ids = np.unique(tracks.track_id, return_inverse=True)
        tracks = tracks.assign(track_id=("record", new_track_ids + current_track_id))

        current_track_id = tracks.track_id.values.max() + 1

        if n == 0:
            all_tracks = tracks
        elif len(tracks.time) > 0:
            all_tracks = xr.concat([all_tracks, tracks], dim="record")
        else:
            warnings.warn(f"{fname} contains no matching tracks")

    return all_tracks


def tidy_track_metadata(tracks, plevs):
    tracks["pressure"] = ("pressure", plevs)
    tracks = tracks.set_coords("pressure")

    vorticity = np.zeros([tracks.sizes["record"], tracks.sizes["pressure"]])
    vorticity_lon = np.zeros_like(vorticity)
    vorticity_lat = np.zeros_like(vorticity)
    for n, plev in enumerate(tracks.pressure.values):
        name = f"vorticity{int(plev)}hPa"
        vorticity[:, n] = tracks[name].values
        vorticity_lon[:, n] = tracks[name + "_lon"].values
        vorticity_lat[:, n] = tracks[name + "_lat"].values

        tracks = tracks.drop_vars([name, name + "_lon", name + "_lat"])

    tracks["relative_vorticity"] = (["record", "pressure"], vorticity)
    tracks["relative_vorticity_lon"] = (["record", "pressure"], vorticity_lon)
    tracks["relative_vorticity_lat"] = (["record", "pressure"], vorticity_lat)

    return tracks


if __name__ == '__main__':
    parse_docopt_arguments(main, __doc__)
