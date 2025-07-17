"""Add cyclone phase space parameters to ERA5 tracks

Subset by year and month to be able to parallelise

Usage:
    add_cps.py <filename> [--year=<year>] [--month=<month>]
    add_cps.py  (-h | --help)

Arguments:
    <filename>
    --year=<year>
    --month=<month>

Options:
    -h --help        Show help

"""
import iris
import pandas as pd
from tqdm import tqdm
import xarray as xr

from twinotter.util.scripting import parse_docopt_arguments

import huracanpy

from huracan import cps


def main(filename, year=None, month=None):
    year = int(year)
    month = int(month)

    # Initial sift of tracks with find_tracks.py
    tracks = huracanpy.load(filename)

    # Subset year and month by first track point
    genesis = tracks.hrcn.get_gen_vals()
    track_ids = genesis.track_id[
        (genesis.time.dt.year == year) &
        (genesis.time.dt.month == month)
    ]
    tracks = tracks.hrcn.sel_id(track_ids)
    tracks = tracks.hrcn.add_azimuth(centering="adaptive")

    # Loop over time and re-sort at the end to avoid having to reload ERA5 data
    all_points = []
    time_groups = tracks.groupby("time")
    for time, points in tqdm(time_groups):
        time = pd.to_datetime(time)
        z_t = get_geopotential_height(time)

        bmax, b, vtl, vtu = [], [], [], []
        for lon, lat, angle in zip(
            points.lon.values, points.lat.values, points.azimuth.values
        ):
            bmax.append(cps.cps_b(z_t, 900, 600, lon, lat, radius=500))
            b.append(cps.cps_b(z_t, 900, 600, lon, lat, radius=500, angle=[angle]))
            vtl.append(cps.cps_vt(z_t, 900, 600, lon, lat, radius=500))
            vtu.append(cps.cps_vt(z_t, 600, 300, lon, lat, radius=500))

        points["cps_bmax"] = ("record", bmax)
        points["cps_b"] = ("record", b)
        points["cps_vtl"] = ("record", vtl)
        points["cps_vtu"] = ("record", vtu)

        all_points.append(points)

    result = xr.concat(all_points, dim="record").sortby("track_id")

    huracanpy.save(result, filename.replace(".nc", f"{year}_{month:02d}.nc"))


def get_geopotential_height(time):
    fname = f"/gws/nopw/j04/huracan/data/ERA5_CPS/z_900-300hPa_{time.year}{time.month:02d}_6h_1deg.nc"

    return iris.load_cube(
        fname,
        iris.Constraint(
            name="geopotential_height",
            time=lambda cell: cell.point == time
        )
    )


if __name__ == '__main__':
    parse_docopt_arguments(main, __doc__)
