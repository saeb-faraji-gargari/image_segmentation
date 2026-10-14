import numpy.typing as npt
from skimage.measure import find_contours


def contourline_segmentation(
    raster: npt.NDArray,
    level: float,
) -> list[npt.NDArray]:
    """
    Extract contour lines from a raster at a specified value.

    Parameters
    ----------
    raster : npt.NDArray
        Two-dimensional raster without NaN values. Missing values
        should be replaced with zero.
    level : float
        Raster value at which to extract the contours.

    Returns
    -------
    list[npt.NDArray]
        List of contour arrays. Each array has shape (N, 2),
        with rows and columns representing pixel coordinates.
    """
    contours = find_contours(raster, level=level)

    return contours
