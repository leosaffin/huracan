from tqdm import tqdm
import xarray as xr

import huracanpy


def main():
    for model_year in range(2015, 2020 + 1):
        if model_year == 2015:
            months = list(range(5, 12 + 1))
        else:
            months = list(range(1, 12 + 1))

        for month in tqdm(months):
            tracks = huracanpy.load(f"hindcast_tracks_NATL_TC_{model_year}_{month}.nc")

            if month == months[0]:
                all_tracks = tracks
            else:
                tracks.track_id[:] = tracks.track_id + all_tracks.track_id.max()
                all_tracks = xr.concat([all_tracks, tracks], dim="record")

        huracanpy.save(all_tracks, f"ECMWF-HINDCASTS_NATL_TCs_{model_year}.nc")


if __name__ == '__main__':
    main()
