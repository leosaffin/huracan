from itertools import groupby

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
import iris
from iris.analysis import MEAN, MAX, MIN
from iris.analysis.cartography import wrap_lons, get_xy_grids
from iris.coords import AuxCoord
from iris.cube import Cube, CubeList
from iris.util import broadcast_to_shape
from cartopy.geodesic import Geodesic

from irise.grid import _within_distance, area_weights_within_distance


geodesic = Geodesic()


def cps_b(
    z: Cube,
    p1: float,
    p2: float,
    lon: float,
    lat: float,
    radius: float = 500,
    angle: [ArrayLike, float] = np.arange(0, 180, 10)
) -> float:
    """Cyclone Phase Space Thermal Asymmetry (B)

    Parameters
    ----------
    z
        Geopotential height
    p1, p2
        Two pressure levels for calculating dZ
    lon, lat
        Location of the cyclone centre (in degrees)
    radius
        The radius (in km) to average around the cyclone
    angle

    Returns
    -------
    The B parameter
    """
    z = z.extract(iris.Constraint(air_pressure=[p2, p1]))

    z_left, z_right = split_cyclone_data(np.array([lon, lat]), angle, z)

    # dz_left[bearing, pressure_level, latitude, longitude]
    dz_left = z_left[:, 1, :, :] - z_left[:, 0, :, :]
    dz_right = z_right[:, 1, :, :] - z_right[:, 0, :, :]

    weights_ = area_weights_within_distance(dz_left, lon, lat, radius)
    weights = weights_ * ~np.isnan(dz_left.data)
    dz_left.data[np.isnan(dz_left.data)] = 0.0
    dz_left = dz_left.collapsed(["latitude", "longitude"], MEAN, weights=weights)

    weights_ = area_weights_within_distance(dz_right, lon, lat, radius)
    weights = weights_ * ~np.isnan(dz_right.data)
    dz_right.data[np.isnan(dz_right.data)] = 0.0
    dz_right = dz_right.collapsed(["latitude", "longitude"], MEAN, weights=weights)

    return np.abs((dz_left - dz_right).data).max()


def cps_vt(
    z: Cube, p1: float, p2: float, lon: float, lat: float, radius: float = 500
) -> float:
    """Cyclone Phase Space Warm Core criteria

    Parameters
    ----------
    z
        Geopotential height
    p1, p2
        Pressure levels for gradient
    lon, lat
        Cyclone centre location
    radius
        Distance (in km) from cyclone to include in diagnostics

    Returns
    -------

    """
    z = z.extract(iris.Constraint(air_pressure=lambda x: p2 <= x <= p1))

    mask = 1 - _within_distance(z, lon, lat, radius)
    mask = broadcast_to_shape(mask, z.shape, [1, 2]) | np.isnan(z.data)

    z_max = z.copy(data=np.ma.masked_where(mask, z.data))
    z_max = z_max.collapsed(["latitude", "longitude"], MAX)

    z_min = z.copy(data=np.ma.masked_where(mask, z.data))
    z_min = z_min.collapsed(["latitude", "longitude"], MIN)

    lnp = np.log(z.coord("air_pressure").points * 100)
    delta_z = (z_max - z_min).data

    return linregress(lnp, delta_z).slope


def split_cyclone_data(
    centre: np.ndarray, bearings, field: iris.cube.Cube
) -> tuple[Cube, Cube]:
    """
    Function for splitting a cyclone into two halves.

    Parameters
    ----------
    centre
        Cyclone centre location (lon, lat)
    bearings
        numpy array of bearings
    field
        iris cube

    Returns
    -------
    Outputs two iris cubes containing data on left and right
    side of the cyclone (on same grid as field, but with some data set to
    NaNs).
    """
    # Make sure bearing varies between -180 and 180
    bearings = wrap_lons(np.asarray(bearings), base=-180, period=360)

    # A list of every lat, lon grid point
    xpoints, ypoints = get_xy_grids(field)
    xypoints = np.array([xpoints.flatten(), ypoints.flatten()]).T

    # Finds bearing between xcentre and every other point on lat, lon grid
    bearing_general = geodesic.inverse(centre, xypoints)[:, -1]
    bearing_general = np.reshape(bearing_general, xpoints.shape)

    left_thetas = CubeList()
    right_thetas = CubeList()
    for bearing in bearings:
        # Iris cubes for left and right half of cyclone
        left_theta = field.copy()
        right_theta = field.copy()

        bearing_dc = AuxCoord(bearing, long_name="bearing", units="degrees")
        left_theta.add_aux_coord(bearing_dc)
        right_theta.add_aux_coord(bearing_dc)

        # Set data on the right half of cyclone to NaN
        right_side = (
            (bearing_general - bearing < 180) & (bearing_general - bearing >= 0)
        ) | (bearing_general - bearing < -180)

        left_theta.data[..., right_side] = np.nan
        right_theta.data[..., ~right_side] = np.nan

        left_thetas.append(left_theta)
        right_thetas.append(right_theta)

    return left_thetas.merge_cube(), right_thetas.merge_cube()


def is_tropical_cyclone(
        b: [ArrayLike, None],
        vtl: [ArrayLike, None],
        vtu: [ArrayLike, None],
        *,
        filter_size: [int, None] = None,
        b_threshold: [float, None] = 10,
        vtl_threshold: [float, None] = 0,
        vtu_threshold: [float, None] = 0,
) -> ArrayLike:
    """Identify where a track is a tropical cyclone by the cyclone phase space
    definition (warm core and symmetric)

    Default thresholds are using North Atlantic definitions from table 1 in
    https://www.sciencedirect.com/science/article/pii/S2225603223000516

    Parameters
    ----------
    b
        Cyclone phase space asymmetry
    vtl
        Cyclone phase space low-level warm core
    vtu
        Cyclone phase space upper-level warm core
    filter_size
        Length (in timesteps) of the uniform filter to apply to the cyclone phase space
        parameters. If None, don't apply filter

    b_threshold
        The threshold of the asymmetry parameter, below which is considered to be a
        tropical cyclone

    vtl_threshold
        The threshold of the low-level warm-core parameter, above which is considered
        to be a tropical cyclone

    vtu_threshold
        The threshold of the upper-level warm-core parameter, above which is considered
        to be a tropical cyclone

    Returns
    -------
    True where the CPS criteria for a tropical cyclone is achieved, False otherwise

    """
    if (
            (b is None and vtl is None and vtu is None) or
            (b_threshold is None and vtl_threshold is None and vtu_threshold is None)
    ):
        raise ValueError("Need to pass at least one variable and threshold")

    if filter_size is not None:
        if b_threshold is not None and b is not None:
            b = uniform_filter1d(b, size=filter_size, mode="nearest")
        if vtl_threshold is not None and vtl is not None:
            vtl = uniform_filter1d(vtl, size=filter_size, mode="nearest")
        if vtu_threshold is not None and vtu is not None:
            vtu = uniform_filter1d(vtu, size=filter_size, mode="nearest")

    condition = []
    if b_threshold is not None and b is not None:
        condition.append(b <= b_threshold)
    if vtl_threshold is not None and vtl is not None:
        condition.append(vtl > vtl_threshold)
    if vtu_threshold is not None and vtu is not None:
        condition.append(vtu > vtu_threshold)

    if len(condition) == 1:
        return condition[0]
    else:
        is_tc = condition[0]
        for other_condition in condition[1:]:
            is_tc = is_tc & other_condition

    return is_tc


def nature(
    b,
    vtl,
    vort,
    is_tc,
    *,
    b_threshold: [float, None] = 10,
    vtl_threshold: [float, None] = 0,
    vort_threshold: [float, None] = 6,
    min_count: [int, None] = 4,
):
    # First guess nature based on TC identification and quadrant of CPS
    nature = np.zeros(len(vort), dtype="U2")
    nature[(b > b_threshold) & (vtl <= vtl_threshold)] = "EC"
    nature[(b > b_threshold) & (vtl > vtl_threshold)] = "WS"
    nature[(b <= b_threshold) & (vtl > vtl_threshold)] = "TS"
    nature[is_tc == 1] = "TC"
    nature[vort < vort_threshold] = "Vo"
    nature[nature == ""] = "ET"

    # Any Warm core/symmetric periods adjacent to TC are also TC
    nature_consecutive = [(k, sum(1 for _ in g)) for k, g in groupby(nature)]
    idx = 0
    for m, (nat, count) in enumerate(nature_consecutive):
        if nat == "TS":
            # Allow for <1 day excursions between TC-TS
            idx_start = max(0, idx - min_count)
            idx_end = min(len(nature), idx + count + min_count)
            if (nature[idx_start:idx_end] == "TC").any():
                nature[idx:idx + count] = "TC"

            else:
                nature[idx:idx + count] = "WS"

        idx += count

    # Smooth out any shorter than 1 day excursions
    nature_consecutive = [(k, sum(1 for _ in g)) for k, g in groupby(nature)]
    idx = nature_consecutive[0][1]
    for m in range(1, len(nature_consecutive) - 1):
        nat, count = nature_consecutive[m]
        if count < min_count:
            if nature_consecutive[m - 1][0] == nature_consecutive[m + 1][0]:
                new_nat = nature_consecutive[m - 1][0]
                nature[idx:idx + count] = new_nat
                nature_consecutive[m] = (new_nat, count)

        idx += count

    # Reclassify TC->WS->EC as TC->ET->EC
    nature_consecutive = [(k, sum(1 for _ in g)) for k, g in groupby(nature)]
    idx = nature_consecutive[0][1]
    for m in range(1, len(nature_consecutive) - 1):
        nat, count = nature_consecutive[m]
        if nat == "WS":
            if (
                    nature_consecutive[m - 1][0] == "TC" and
                    nature_consecutive[m + 1][0] == "EC"
            ):
                nature[idx:idx + count] = "ET"

        idx += count

    return nature
