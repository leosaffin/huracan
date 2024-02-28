import datetime

import numpy as np

from pylagranto import caltra
from pylagranto.datasets import ERA5

from huracan.nawdex import nawdex_start, nawdex_end


def main():
    # Calculate 72-hour back trajectories from 6-hourly ERA5 data on pressure levels for
    # each time within the NAWDEX campaign
    filename = "../data/era5/era5_*levs_{year:04d}{month:02d}{day:02d}.nc"
    step = datetime.timedelta(hours=6)

    trainp = []
    for x in range(-100, 20):
        for y in range(0, 90):
            for p in range(50, 1050, 50):
                trainp.append([x, y, p*100])
    trainp = np.array(trainp)

    time = nawdex_start
    while time <= nawdex_end:
        times = [time - datetime.timedelta(hours=dt) for dt in range(0, 72+1, 6)]
        mapping = {t: filename.format(year=t.year, month=t.month, day=t.day) for t in times}
        datasource = ERA5(mapping)

        # Calculate the trajectories
        traout = caltra.caltra(trainp, times, datasource, fbflag=-1, tracers=["air_potential_temperature"])

        # Save the trajectories
        traout.save(f"nawdex_3d_back_trajectories_{time.strftime('%Y%m%dT%H%M')}.pkl")

        time += step


if __name__ == '__main__':
    main()
