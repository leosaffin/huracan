"""

Usage:
    compare_ibtracs.py <filename_tracks> <filename_ibtracs> <matching_table_filename> <plot_path>
    compare_ibtracs.py  (-h | --help)

Arguments:
    <filename_tracks>
    <filename_ibtracs>
    <matching_table_filename>
    <plot_path>

Options:
    -h --help        Show help
"""

from datetime import timedelta
from itertools import groupby
import pathlib
import warnings

from cartopy.crs import EqualEarth, Geodetic, PlateCarree
import huracanpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from twinotter.util.scripting import parse_docopt_arguments

from huracan.cps._cps import nature

projection = EqualEarth
transform = Geodetic()

extents = dict(
    NATL=[-120, 60, 0, 90],
    MED=[-120, 60, 0, 90],
    ENP=[-180, 0, 0, 90],
    CP=[120, 300, 0, 90],
    WNP=[60, 240, 0, 90],
    NI=[0, 180, 0, 90],
    SI=[0, 180, -90, 0],
    AUS=[60, 240, -90, 0],
    SP=[120, 300, -90, 0],
    SA=[-120, 60, -90, 0],
)

# Color for each cyclone category
colors = {"EC": "C0", "TC": "C1", "ET": "C2", "WS": "C3", "Vo": "C7"}

# Layout of the figure for plt.subplot_mosaic
mosaic = """
    lAA
    lAA
    lAA
    mAA
    mAA
    mAA
    000
    x1y
    B1C
    B1C
    D1E
    D1E
    F1G
    F1G
"""


def main(filename_tracks, filename_ibtracs, matching_table_filename, plot_path):
    tracks = huracanpy.load(filename_tracks)
    ibtracs = huracanpy.load(filename_ibtracs)
    superbt = huracanpy.load(source="superbt").rename(
        tccode="nature", vmax="wind", pmin="slp"
    )
    table = pd.read_parquet(matching_table_filename)
    # Ignore entries only in nolat-tcident and nothing else
    table = table[(table.id_ibtracs != "-1") | table.in_H2017 | table.in_WCS]

    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(exist_ok=True)

    ibtracs.wind.attrs["units"] = "kts"
    ibtracs["wind"] = ibtracs.wind.metpy.convert_units("m s-1").metpy.dequantify()
    superbt["wind"] = superbt.wind.metpy.convert_units("m s-1").metpy.dequantify()

    # Specific tracks for WCSI paper
    _main(tracks, ibtracs, superbt, table, plot_path)


def plot_all(tracks, ibtracs, table, plot_path):
    pass


def _main(tracks, ibtracs, superbt, table, path):
    # True positive Iris
    table_ = table[table.id_ibtracs == "1995235N13311"]
    track = tracks.hrcn.sel_id(table_.id_era5)
    ib = ibtracs.hrcn.sel_id(table_.id_ibtracs)
    fig, ax = track_overview(track, ib)
    ax["A"].set_title("Iris (1995)")
    plt.savefig(path / "iris_1995.pdf")
    plt.close()

    # False negative Earl
    table_ = table[table.id_ibtracs == "1986254N22309"]
    track = tracks.hrcn.sel_id(table_.id_era5)
    ib = ibtracs.hrcn.sel_id(table_.id_ibtracs)
    fig, ax = track_overview(track, ib)
    ax["A"].set_title("Earl (1986)")
    plt.savefig(path / "earl_1986.pdf")
    plt.close()

    # Invest
    table_ = table[table.id_superbt == "b8l.2013"]
    track = tracks.hrcn.sel_id(table_.id_era5)
    ib = superbt.hrcn.sel_id(table_.id_superbt)
    fig, ax = track_overview(track, ib)
    ax["A"].set_title(f"Invest b8l (2013)")
    plt.savefig(path / f"invest_b8l_2013.pdf")
    plt.close()

    # False positives
    table_ = table[np.isin(table.id_era5, [67491, 67214, 195394])]
    for n, row in tqdm(table_.iterrows()):
        track = tracks.hrcn.sel_id(row.id_era5)
        fig, ax = track_overview(track)
        ax["A"].set_title(f"False Positive {row.id_era5} ({row.year})")
        plt.savefig(path / f"false-positive_{row.id_era5}_{row.year}.pdf")
        plt.close()


def track_overview(track=None, ib=None):
    basin = guess_basin(track, ib)
    fig, axes = create_figure(basin)
    lines_axes = []

    if track is not None:
        # Calculate track parameters
        time = track.time
        tmin, tmax = time.min(), time.max()

        # Cyclone phase space
        b = uniform_filter1d(np.abs(track.cps_b), size=5, mode="nearest")
        vtl = uniform_filter1d(track.cps_vtl, size=5, mode="nearest")
        vtu = uniform_filter1d(track.cps_vtu, size=5, mode="nearest")

        vo850 = track.relative_vorticity.sel(pressure=850)
        vo200 = track.relative_vorticity.sel(pressure=200)

        vorticity = uniform_filter1d(vo850, size=5, mode="nearest")
        dvort = vo850 - vo200
        nat = nature(b, vtl, vorticity, track.is_tc.values, b_threshold=15)

        idx = 0
        for nat_, npoints in [(a[0], sum([1 for _ in a[1]])) for a in groupby(nat)]:
            track.isel(
                record=slice(idx, min(idx + npoints + 1, len(track.time)))
            ).hrcn.plot_fancyline(
                colors=colors[nat_], ax=axes["A"], linewidths=4,
            )
            idx += npoints

    # IBTrACS overlay
    if ib is not None:
        ib_tc = ib.isel(record=np.where(ib.nature == "TS")[0])
        ib_not_tc = ib.isel(record=np.where(ib.nature != "TS")[0])
        time = ib.time
        if track is None:
            tmin, tmax = time.min(), time.max()
        else:
            tmin = min(tmin, time.min())
            tmax = max(tmax, time.max())

        idx = 0
        istc_ib = ib.nature == "TS"
        for nat_, npoints in [(a[0], sum([1 for _ in a[1]])) for a in groupby(istc_ib)]:
            if nat_:
                color = "C4"
            else:
                color = "C5"

            ib_ = ib.isel(
                record=slice(idx, min(idx + npoints + 1, len(ib.time)))
            )

            if len(ib_.record) > 1:
                ib_.hrcn.plot_fancyline(
                    colors="w", ax=axes["A"], linewidths=2
                )
                ib_.hrcn.plot_fancyline(
                    colors=color, ax=axes["A"]
                )

            idx += npoints

    for cond, color in [
        ("Tropical", "C1"),
        ("Extratropical", "C0"),
        ("Transition", "C2"),
        ("Warm Seclusion", "C3"),
        ("Vortex", "C7"),
    ]:
        axes["l"].fill_betweenx(
            [np.nan, np.nan], np.nan, np.nan, color=color, alpha=0.4, label=cond
        )
    axes["l"].legend(title="ERA5 Nature", bbox_to_anchor=[0.75, 0.75])

    for cond, color in [
        ("Tropical Storm", "C4"),
        ("Other", "C5"),
    ]:
        axes["m"].fill_betweenx(
            [np.nan, np.nan], np.nan, np.nan, color=color, alpha=0.4, label=cond
        )
    if ib is not None:
        axes["m"].legend(title="IBTrACS Nature", bbox_to_anchor=[0.75, 0.75])

    axes["B"].set(
        ylim=(950, 1025),
        yticks=[950, 975, 1000, 1025],
        ylabel="MSLP (hPa)",
    )

    if track is not None:
        axes["D"].set(
            ylim=(0, 50),
            yticks=[0, 6, 25, 50],
            ylabel="$\\xi_\\mathrm{850hPa}$\n(10$^{-5}$ s$^{-1}$)"
        )
    axes["F"].set(
        ylim=(0, 80),
        yticks=[0, 40, 80],
        ylabel=r"$v_\mathrm{max, 10m}$ (m s$^{-1}$)"
    )

    if track is not None:
        axes["C"].set(
            ylim=(0, 120),
            yticks=[0, 15, 60, 120],
            ylabel="Asymmetry (B)",
        )
        axes["E"].set(
            ylim=(-300, 300),
            yticks=[-300, 0, 300],
            ylabel=r"Thermal Wind (V$_\mathrm{T}$)"
        )
        axes["G"].set(
            ylim=(-20, 20),
            yticks=[-20, 0, 6, 20],
            ylabel=(
                "$\\xi_\\mathrm{850hPa} - \\xi_\\mathrm{200hPa}$\n(10$^{-5}$ s$^{-1}$)"
            )
        )
    else:
        axes["C"].remove()

    axes["x"].set(ylim=(0, 1), title="Intensity")
    axes["y"].set(ylim=(0, 1), title="Structure")

    for ax_label in ["B", "C", "D", "E", "F", "G", "x", "y"]:
        axes[ax_label].set(xlim=(tmin, tmax), clip_on=False)

        for direction in ["top", "right"]:
            axes[ax_label].spines[direction].set_visible(False)

    for ax_label in ["x", "y", "B", "C", "D", "E"]:
        axes[ax_label].spines["bottom"].set_visible(False)
        axes[ax_label].get_xaxis().set_ticks([])

    for ax_label in ["D", "E"]:
        axes[ax_label].yaxis.tick_right()
        axes[ax_label].yaxis.set_label_position("right")
        axes[ax_label].spines["left"].set_visible(False)
        axes[ax_label].spines["right"].set_visible(True)

    for ax_label in ["x", "y"]:
        axes[ax_label].spines["left"].set_visible(False)
        axes[ax_label].get_yaxis().set_ticks([])

    fig.autofmt_xdate()

    # Fill for different criteria
    for ax_label in ["B", "C", "D", "E", "F", "G"]:
        ax = axes[ax_label]
        y = ax.get_ylim()

        if track is not None:
            for cond, color in [
                ("EC", "C0"),
                ("TC", "C1"),
                ("ET", "C2"),
                ("WS", "C3"),
                ("Vo", "C7"),
            ]:
                track_ = track.isel(record=np.where(nat == cond)[0])
                times_ = pd.to_datetime(track_.time)
                for time_ in times_:
                    ax.fill_betweenx(
                        [y[0], y[1]],
                        time_ - timedelta(hours=3),
                        time_ + timedelta(hours=3),
                        color=color,
                        ec=None,
                        alpha=0.4,
                    )
    for ax_label in ["x", "y"]:
        ax = axes[ax_label]
        y = ax.get_ylim()
        # IBTrACS Tropical Storm
        if ib is not None:
            times_tc_ib = pd.to_datetime(ib_tc.time)
            for time_ in times_tc_ib:
                ax.fill_betweenx(
                    [y[0], y[1]],
                    time_ - timedelta(hours=3),
                    time_ + timedelta(hours=3),
                    color="C4",
                    ec=None,
                    alpha=0.4,
                )
            times_tc_ib = pd.to_datetime(ib_not_tc.time)
            for time_ in times_tc_ib:
                ax.fill_betweenx(
                    [y[0], y[1]],
                    time_ - timedelta(hours=3),
                    time_ + timedelta(hours=3),
                    color="C5",
                    ec=None,
                    alpha=0.4,
                )

    if track is not None:
        # Intensity
        lb0, lb1, lb2 = intensityplot(
            axes["B"],
            axes["D"],
            axes["F"],
            track.time,
            track.mslp,
            vorticity,
            track.vmax10m
        )
        lines_axes += [(lb0, axes["B"]), (lb1, axes["D"]), (lb2, axes["F"])]

        # CPS
        lc0, lc1, lc2 = cpsplot(
            axes["C"], axes["E"], axes["G"], track.time, b, vtl, vtu, dvort
        )
        lines_axes += [(lc0, axes["C"]), (lc1, axes["E"]), (lc2, axes["G"])]

        axes["E"].legend(bbox_to_anchor=[0, 0.9])

    if ib is not None:
        lb0, lb1, lb2 = intensityplot(
            axes["B"],
            axes["D"],
            axes["F"],
            ib.time,
            ib.slp,
            np.full_like(ib.slp, np.nan),
            ib.wind,
            linestyle="--",
            label="IBTrACS"
        )

        if track is None:
            lines_axes += [(lb0, axes["B"]), (lb1, axes["D"]), (lb2, axes["F"])]

    if ib is not None:
        axes["B"].legend(bbox_to_anchor=[1, 1])

    for line, axis in lines_axes:
        axis.yaxis.label.set_color(line[0].get_color())
        axis.tick_params(axis="y", colors=line[0].get_color())

    axes["A"].text(0, 1.05, "(a)", transform=axes["A"].transAxes)
    axes["x"].text(0, 1.05, "(b)", transform=axes["x"].transAxes)
    axes["y"].text(0, 1.05, "(c)", transform=axes["y"].transAxes)

    axes["A"].set_extent(extents[basin], crs=PlateCarree())

    return fig, axes


def guess_basin(track, ib):
    if track is not None:
        basin = track.hrcn.get_basin().values[track.vorticity.argmax()]
    else:
        try:
            basin = ib.hrcn.get_basin().values[ib.wind.argmax()]
        except ValueError:
            try:
                basin = ib.hrcn.get_basin().values[ib.slp.argmin()]
            except ValueError:
                basin = ib.hrcn.get_basin().values[0]

    return basin


def create_figure(basin):
    central_lon = 0.5 * (extents[basin][0] + extents[basin][1])
    fig, axes = plt.subplot_mosaic(
        mosaic, figsize=(8, 9), per_subplot_kw=dict(A=dict(
            projection=projection(central_longitude=central_lon)
        ))
    )

    axes["l"].set_axis_off()
    axes["m"].set_axis_off()

    axes["0"].remove()
    axes["1"].remove()
    plt.subplots_adjust(wspace=0, hspace=0)

    axes["A"].set_extent(extents[basin], crs=PlateCarree())
    axes["A"].stock_img()
    axes["A"].coastlines()
    axes["A"].gridlines(
        xlocs=range(extents[basin][0], extents[basin][1] + 1, 30),
        ylocs=range(extents[basin][2], extents[basin][3], 15),
        draw_labels=["left", "bottom"],
        color="w",
        zorder=-1,
    )

    return fig, axes


def intensityplot(
    ax, axt1, axt2, time, mslp, vorticity, wind, linestyle="-", label="ERA5"
):
    # Intensity
    l0 = ax.plot(time, mslp, "k", linestyle=linestyle, label=label)
    l1 = axt1.plot(time, vorticity, "C4", linestyle=linestyle)
    l2 = axt2.plot(time, wind, "C2", linestyle=linestyle)

    showlimit(axt1, time, vorticity, 6, color="C4")

    return l0, l1, l2


def cpsplot(ax, axt1, axt2, time, b, vtl, vtu, dvort):
    l0 = ax.plot(time, b, "-C0")
    showlimit(ax, time, b, 15, above=False, color="C0")

    l1 = axt1.plot(time, vtl, "-C3", label=r"V$_\mathrm{T}^\mathrm{L}$")
    l2 = axt1.plot(time, vtu, "--C3", label=r"V$_\mathrm{T}^\mathrm{U}$")
    showlimit(axt1, time, vtl, 0, color="C3")
    showlimit(axt1, time, vtu, 0, color="C3")

    l3 = axt2.plot(time, dvort, "-C5")
    showlimit(axt2, time, dvort, 6, color="C5")

    return l0, l1, l3


def combined_legend(ax, lines):
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)


def showlimit(ax, x, y, ylim, above=True, color="k"):
    ax.axhline(ylim, color=color)

    if above:
        y_masked = np.where(y >= ylim, y, np.nan)
    else:
        y_masked = np.where(y < ylim, y, np.nan)

    ax.fill_between(x, y_masked, ylim, color=color, alpha=0.25)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parse_docopt_arguments(main, __doc__)
