from itertools import groupby

import numpy as np
import pandas as pd
from tqdm import tqdm

import huracanpy

from huracan.cps import is_tropical_cyclone


def main():
    summary = pd.DataFrame(
        data=dict(
            track_id=[],
            storm_start=[],
            storm_end=[],
            origin_lat=[],
            origin_lon=[],
            end_lat=[],
            end_lon=[],
            is_wc_sym=[],
            is_tc=[],
        )
    )

    tracks = huracanpy.load(f"ERA5_tracks_CPS_NATL_990hPa.nc")

    tracks = tracks.hrcn.add_is_ocean()

    for track_id, track in tqdm(tracks.groupby("track_id")):
        # Only storms that are tropical cyclones at some point in their lifecycle
        b = np.ma.masked_where(np.isnan(track.cps_b), track.cps_b)
        vtl = np.ma.masked_where(np.isnan(track.cps_vtl), track.cps_vtl)
        vtu = np.ma.masked_where(np.isnan(track.cps_vtu), track.cps_vtu)
        tc = is_tropical_cyclone(b, vtl, vtu, filter_size=4)
        tc = (
                tc
                & (track.relative_vorticity.isel(pressure=0) >= 6)
                & (track.basin == "NATL")
                & track.is_ocean
        )

        category_consecutive = [
            (k, sum(1 for i in g)) for k, g in groupby(tc.values)
        ]
        is_wc_sym = False
        for category, count in category_consecutive:
            if category and count > 1:
                is_wc_sym = True

        times = pd.to_datetime(track.time)
        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    [
                        dict(
                            track_id=track.track_id.values[0],
                            storm_start=times[0],
                            storm_end=times[-1],
                            origin_lat=track.lat.data[0],
                            origin_lon=track.lon.data[0],
                            end_lat=track.lat.data[-1],
                            end_lon=track.lon.data[-1],
                            is_wc_sym=is_wc_sym,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    summary.to_parquet(f"ERA5_tracks_CPS_NATL_990hPa_summary.parquet")


if __name__ == "__main__":
    main()
