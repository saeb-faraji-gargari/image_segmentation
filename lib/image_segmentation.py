import numpy as np
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


def contourline_segmentation_with_padding(
    raster: npt.NDArray,
    level: float,
) -> list[npt.NDArray]:
    """
    Extract contour lines from a raster using zero-padding at its boundaries.

    The raster is padded with one pixel of zeros on all sides before
    contour extraction. This allows contours intersecting the original
    raster boundaries to extend into the padded region and potentially
    form closed contours.

    The padding offset is removed from the resulting contour coordinates
    to preserve their positions relative to the original raster.

    Parameters
    ----------
    raster : npt.NDArray
        Two-dimensional raster without NaN values.
    level : float
        Raster value at which to extract the contours.

    Returns
    -------
    list[npt.NDArray]
        List of contour arrays. Each array has shape (N, 2),
        with rows and columns representing pixel coordinates
        relative to the original raster.
    """
    padded_raster = np.pad(
        raster,
        pad_width=1,
        mode="constant",
        constant_values=0,
    )

    contours = find_contours(padded_raster, level=level)

    contours = [contour - 1 for contour in contours]

    return contours
