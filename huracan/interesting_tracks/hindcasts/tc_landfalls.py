"""

Usage:
    tc_landfalls.py <filename_in> [<filename_out>]
    tc_landfalls.py  (-h | --help)

Arguments:
    <filename_in>
        The tracks file to load
    <filename_out>
        The filename to save the filtered tracks.


Options:
    -h --help        Show help

"""

import huracanpy
import numpy as np
from parse_docopt import parse_docopt
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
import xarray as xr


def main(filename_in, filename_out=None):
    if filename_out is None:
        filename_out = filename_in.replace(".nc", "_TC-landfall.nc")
    print(filename_in, filename_out)

    tracks = huracanpy.load(filename_in)
    tracks["iseurope"] = tracks.hrcn.get_basin(convention="Sainsbury2022MWR")

    tc_landfalls = []

    for track_id, track in tqdm(tracks.groupby("track_id")):
        # Tracks are initialised as TC, so just need to subset those that retain a
        # warm core, symmetric structure all the way to landfall
        track_ = track.isel(record=np.where(track.time.dt.hour % 12 == 0)[0])

        b = uniform_filter1d(np.abs(track_.cps_b), size=3, mode="nearest")
        vtl = uniform_filter1d(track_.cps_vtl, size=3, mode="nearest")
        idx_landfall = np.where(track_.basin == "Europe")[0][0]

        if ((b[0:idx_landfall] < 15) & (vtl[0:idx_landfall] > 0)).all():
            tc_landfalls.append(track_)

    if len(tc_landfalls) > 0:
        tc_landfalls = xr.concat(tc_landfalls, dim="record")
        huracanpy.save(tc_landfalls, filename_out)
    else:
        print("No TC landfalls")


if __name__ == '__main__':
    main(**parse_docopt(__doc__))
