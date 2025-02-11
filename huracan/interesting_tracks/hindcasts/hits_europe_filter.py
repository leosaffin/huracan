"""

Usage:
    find_tracks.py <filename_in> [<filename_out>]
    find_tracks.py  (-h | --help)

Arguments:
    filename_in
        The tracks file to load
    filename_out
        The filename to save the filtered tracks.


Options:
    -h --help        Show help

"""

import numpy as np

from twinotter.util.scripting import parse_docopt_arguments
import huracanpy


def main(filename_in, filename_out=None):
    print(filename_in, filename_out)
    tracks = huracanpy.load(filename_in)

    iseurope = huracanpy.info.basin(
        tracks.lon, tracks.lat, convention="Sainsbury2022MWR"
    ) == "Europe"

    subset = tracks.isel(record=np.where(iseurope)[0])
    tracks_ = tracks.isel(record=np.where(
        np.isin(tracks.track_id, np.unique(subset.track_id))
    )[0])

    if filename_out is None:
        filename_out = filename_in.replace(".nc", "_hits-europe.nc")

    huracanpy.save(tracks_, filename_out)


if __name__ == '__main__':
    parse_docopt_arguments(main, __doc__)
