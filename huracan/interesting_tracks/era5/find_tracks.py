"""
Extract all ERA5 tracks on JASMIN that have a pressure minimum below 990hPa over the
North Atlantic ocean, and merge to a single netCDF
"""

import numpy as np
import xarray as xr
from tqdm import tqdm

import huracanpy
from jasmin_tracks import datasets


def main():
    dataset = datasets["ERA5"]
    all_files = list(dataset.find_files())

    current_track_id = 1
    for n, fname in tqdm(enumerate(all_files)):
        tracks = huracanpy.load(
            str(fname), tracker="TRACK", variable_names=dataset.variable_names
        )

        # Filter tracks
        # <=990hPA over North Atlantic Ocean
        tracks["is_ocean"] = (
            huracanpy.utils.geography.get_land_or_ocean(tracks.lon, tracks.lat)
            == "Ocean"
        )
        tracks["basin"] = huracanpy.utils.geography.get_basin(tracks.lon, tracks.lat)

        condition = tracks.is_ocean & (tracks.basin == "NATL") & (tracks.mslp <= 990)
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

    huracanpy.save(all_tracks, "ERA5_tracks_NATL_990hPa.nc")


if __name__ == "__main__":
    main()
