import numpy as np
import numpy.typing as npt
from osgeo import gdal


def read_tif(in_file_path: str) -> npt.NDArray:
    """
    Read the first band of a GeoTIFF file into a NumPy array.

    Converts the band's NoData values to NaN, if a NoData value is defined,
    and prints basic raster statistics.

    Parameters
    ----------
    in_file_path : str
        Path to the input GeoTIFF file.

    Returns
    -------
    raster_without_nan
    npt.NDArray
            Two-dimensional NumPy array containing the raster data, with
            NoData values replaced with zero.

    raster
    npt.NDArray
        Two-dimensional NumPy array containing the raster data, with
        NoData values represented as NaN.

    Raises
    ------
    FileNotFoundError
        If the GeoTIFF file cannot be opened.
    """
    dataset = gdal.Open(in_file_path)

    if dataset is None:
        raise FileNotFoundError(f"Could not open file: {in_file_path}")

    band = dataset.GetRasterBand(1)
    raster = band.ReadAsArray()

    nodata_value = band.GetNoDataValue()

    if nodata_value is not None:
        raster = raster.astype(float)
        raster[raster == nodata_value] = np.nan

    print("Shape:", raster.shape)
    print("Min:", np.nanmin(raster))
    print("Max:", np.nanmax(raster))
    print("Number of NaN:", np.isnan(raster).sum())
    print("Total pixels:", raster.size)
    print("NaN fraction:", np.isnan(raster).mean())

    # # =========================================================
    # # 2. Replace NaN with zero probability
    # # =========================================================

    raster_without_nan = raster.copy()

    raster_without_nan[np.isnan(raster_without_nan)] = 0.0

    return raster_without_nan, raster
