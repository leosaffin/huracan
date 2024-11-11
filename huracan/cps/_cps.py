import numpy as np
from scipy.stats import linregress
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
    z: Cube, p1: float, p2: float, lon: float, lat: float, radius: float = 500
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

    Returns
    -------
    The B parameter
    """
    z = z.extract(iris.Constraint(air_pressure=[p2, p1]))

    bearing = np.arange(0, 180, 10)
    z_left, z_right = split_cyclone_data(np.array([lon, lat]), bearing, z)

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
    bearings = wrap_lons(bearings, base=-180, period=360)

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
