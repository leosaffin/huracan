import subprocess
import datetime

from parse import parse
from tqdm import tqdm

from jasmin_tracks import datasets
from twinotter.util.scripting import parse_docopt_arguments

_YYYYMMDDHH = "{year:04d}{month:02d}{day:02d}{hour:02d}"
_YYYYMMDDHH_model = _YYYYMMDDHH.replace("year", "model_year").replace("day", "model_day")

leap_year_extra_path = (
        f"{_YYYYMMDDHH_model}/{_YYYYMMDDHH}/"
        f"HIND_VOR_VERTAVG_{_YYYYMMDDHH_model}_{_YYYYMMDDHH}" +
        "_{ensemble_member}/"
)


def get_tilts(**kwargs):
    keys = kwargs.keys()
    for key in keys:
        kwargs[key] = int(kwargs[key])

    tilt_args = ["1", "6", "1 850", "2 700", "3 500", "4 400", "5 300", "6 200", "n", "n"]
    dataset = datasets["ECMWF_hindcasts"]
    all_files = list(dataset.find_files(**kwargs))

    for fname in tqdm(all_files):
        if "HIND_VOR_VERTAVG_2016060900_2011060900_10" in str(fname):
            # File has an extra variable so breaks the loop
            continue

        # Fix for leap years
        if "022900_" in str(fname):
            details = parse(str(dataset.fixed_path / leap_year_extra_path / dataset.filename), str(fname))
        else:
            details = dataset.file_details(str(fname))

        start_time = datetime.datetime(
            **{key: details[key] for key in ["year", "month", "day", "hour"]}
        )
        track_id_extra = (
            f"{details['model_year']:04d}_"
            f"{start_time.strftime('%Y%m%d%H%M')}_"
            f"{details['ensemble_member']}_"
        )

        ps = subprocess.Popen(["echo", str(fname)] + tilt_args, stdout=subprocess.PIPE)
        output = subprocess.check_output(["/home/users/kihodges/TRACK-1.5.4/utils/bin/tilt"], stdin=ps.stdout)
        ps.wait()
        subprocess.run(["mv", "tilt.dat", f"/gws/nopw/j04/huracan/data/tracks/tropical_cyclones/tilt/ECMWF_HINDCASTS/hindcast_interesting_tracks_tilts_{track_id_extra[:-1]}.dat"])


if __name__ == "__main__":
    parse_docopt_arguments(get_tilts, __doc__)
