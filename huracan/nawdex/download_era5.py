import cdsapi


def main():
    c = cdsapi.Client()

    year = 2016
    times = ["{:02d}:00".format(n) for n in range(0, 24, 6)]
    area = [90, -100, 0, 20]  # North, West, South, East
    plevs = list(range(50, 1050, 50))
    variables = [
        'geopotential',
        'potential_vorticity',
        'specific_humidity',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'vertical_velocity',
        'vorticity',
    ]

    for month in [9, 10]:
        for day in range(1, 30 + 1):
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': variables,
                    'pressure_level': plevs,
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': times,
                    'area': area,
                },
                f"era5_plevs_{year:04d}{month:02d}{day:02d}.nc")

            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': 'surface_pressure',
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': times,
                    'area': area,
                },
                f"era5_slevs_{year:04d}{month:02d}{day:02d}.nc")


if __name__ == '__main__':
    main()

