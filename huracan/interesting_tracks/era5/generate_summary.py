from itertools import groupby
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import huracanpy
from cyclophaser._determine_periods import process_vorticity

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
            is_weak=[],
            is_wc_sym=[],
            is_tc=[],
        )
    )

    tracks = huracanpy.load(f"ERA5_tracks_CPS_NATL.nc")

    tracks = tracks.hrcn.add_is_ocean()

    for track_id, track in tqdm(tracks.groupby("track_id")):
        is_wc_sym = False
        is_tc = False
        vorticity = track.relative_vorticity.sel(pressure=850)
        is_weak = (vorticity.values < 6).all()

        if not is_weak:
            # Only storms that are tropical cyclones at some point in their lifecycle
            b = np.ma.masked_where(np.isnan(track.cps_b), track.cps_b)
            vtl = np.ma.masked_where(np.isnan(track.cps_vtl), track.cps_vtl)
            tc = is_tropical_cyclone(b, vtl, None, filter_size=4, b_threshold=15)
            tc = (tc & (vorticity >= 6) & (track.basin == "NATL") & track.is_ocean)

            category_consecutive = [
                (k, sum(1 for i in g)) for k, g in groupby(tc.values)
            ]

            for category, count in category_consecutive:
                if category and count >= 4:
                    is_wc_sym = True

            if is_wc_sym:
                result = process_vorticity(
                    vorticity,
                    replace_endpoints_with_lowpass=False,
                    use_filter=False,
                    use_smoothing=16,
                    use_smoothing_twice=8,
                )

                tc = tc & (result[4] > 0)

                category_consecutive = [
                    (k, sum(1 for i in g)) for k, g in groupby(tc.values)
                ]
                for category, count in category_consecutive:
                    if category and count >= 4:
                        is_tc = True

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
                            is_weak=is_weak,
                            is_wc_sym=is_wc_sym,
                            is_tc=is_tc,
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    summary.to_parquet(f"ERA5_tracks_CPS_NATL_summary.parquet")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
