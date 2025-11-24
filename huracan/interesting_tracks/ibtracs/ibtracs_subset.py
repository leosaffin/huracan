import huracanpy
import numpy as np


def main():
    # Load a subset of variables from the full set of IBTrACS data
    ibtracs = huracanpy.load(
        source="ibtracs",
        ibtracs_subset="ALL",
        usecols=[
            "SID",
            "ISO_TIME",
            "LON",
            "LAT",
            "SEASON",
            "BASIN",
            "NAME",
            "NATURE",
            "WMO_WIND",
            "WMO_PRES",
        ],
    ).rename(dict(wmo_wind="wind", wmo_pres="slp"))

    # Only include tracks that start and end within 1940-2022
    start_points = ibtracs.hrcn.get_gen_vals()
    end_points = ibtracs.hrcn.get_apex_vals("time", stat="max")
    track_ids = start_points.track_id[
        (start_points.time.dt.year >= 1940) & (end_points.time.dt.year <= 2024)
    ]
    ibtracs = ibtracs.hrcn.sel_id(track_ids)

    # Only include 6-hourly data
    ibtracs = ibtracs.isel(record=np.where(ibtracs.time.dt.hour % 6 == 0)[0])

    # Only include storms that are labelled as tropical storm for at least one point
    ibtracs = ibtracs.hrcn.trackswhere(
        lambda track: (track.nature == "TS").any() and track.time.size >= 4
    )

    ibtracs.hrcn.save("IBTrACS_6h_1940-2024_Tropical-Storms.nc")


if __name__ == "__main__":
    main()
