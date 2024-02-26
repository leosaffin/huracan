import numpy as np


# Filter criteria
# Apply to a dataframe summary of interesting tracks
def any_point_tropical(df):
    return df.loc[np.logical_not(np.isnan(df.min_pressure_tropics))]


def tropical_origin(df, threshold_latitude=30):
    return df.loc[df.origin_lat < threshold_latitude]


def initialised(df):
    return df.loc[df.storm_start == df.forecast_start]


def generated(df):
    return df.loc[df.storm_start != df.forecast_start]
