"""Add cyclone phase space parameters to ERA5 tracks

Subset by year and month to be able to parallelise

Usage:
    add_cps.py filename [--year=<year>] [--month=<month>]
    add_cps.py  (-h | --help)

Arguments:
    filename=filename
    --year=<year>
    --month=<month>

Options:
    -h --help        Show help

"""

from tqdm import tqdm
from iris.cube import Cube
from iris.coords import DimCoord
import numpy as np
import pandas as pd
import xarray as xr
from metpy.interpolate import interpolate_to_isosurface

from twinotter.util.scripting import parse_docopt_arguments

import jasmin_era5
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
        cubes = jasmin_era5.load(time)
        z_t = get_geopotential_height(cubes, levels=range(90000, 60000 - 1, -5000))
        del cubes

        bmax, b, vtl, vtu = [], [], [], []
        for lon, lat, angle in zip(
            points.lon.values, points.lat.values, points.azimuth.values
        ):
            bmax.append(cps.cps_b(z_t, 900, 600, lon, lat, radius=500))
            b.append(cps.cps_b(z_t, 900, 600, lon, lat, radius=500), angle=angle)
            vtl.append(cps.cps_vt(z_t, 900, 600, lon, lat, radius=500))
            vtu.append(cps.cps_vt(z_t, 600, 300, lon, lat, radius=500))

        points["cps_bmax"] = ("record", bmax)
        points["cps_b"] = ("record", b)
        points["cps_vtl"] = ("record", vtl)
        points["cps_vtu"] = ("record", vtu)

        all_points.append(points)

    result = xr.concat(all_points, dim="record").sortby("track_id")

    huracanpy.save(result, filename.replace(".nc", f"{year}_{month:02d}.nc"))


def get_geopotential_height(cubes, levels=(90000, 60000, 30000)):
    lnsp = cubes.extract_cube("Logarithm of surface pressure")
    temperature = cubes.extract_cube("air_temperature")
    q = cubes.extract_cube("specific_humidity")
    z_s = cubes.extract_cube("geopotential")
    p, p_half = jasmin_era5.p_on_model_levels(lnsp, temperature.shape)
    z = jasmin_era5.height_on_model_levels(p_half, temperature, q, z_s)

    z_p = [interpolate_to_isosurface(p, z, level) for level in levels]
    z_p = Cube(
        data=z_p,
        standard_name="geopotential_height",
        units="m",
        dim_coords_and_dims=[
            (DimCoord(
                points=[level / 100 for level in levels],
                standard_name="air_pressure",
                units="hPa",
            ), 0),
            (temperature.coord("latitude"), 1),
            (temperature.coord("longitude"), 2),
        ],
    )

    # interpolate_to_isosurface doesn't automatically mask out of bounds data. Just sets
    # it to the zero-index value. Mask where this happens
    z_p.data[z_p.data == z[0]] = np.nan

    return z_p


if __name__ == '__main__':
    parse_docopt_arguments(main, __doc__)
