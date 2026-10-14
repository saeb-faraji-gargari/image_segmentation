import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from osgeo import gdal


def plot_numpy_raster(numpy_raster: npt.NDArray) -> None:
    """
    Plot a NumPy raster as a probability map.

    Displays the raster using a grayscale colormap with a fixed
    color scale from 0 to 1.

    Parameters
    ----------
    numpy_raster : npt.NDArray
        Two-dimensional NumPy array containing raster values.
    """
    plt.imshow(
        numpy_raster,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.title("Raster map")
    plt.colorbar(label="Probability")
    plt.tight_layout()
    plt.show()


def plot_tif(in_file_path: str) -> None:
    """
    Read and plot the first band of a GeoTIFF as a probability map.

    Prints basic raster statistics and displays the raster using a
    grayscale colormap with a fixed color scale from 0 to 1.

    Parameters
    ----------
    in_file_path : str
        Path to the input GeoTIFF file.
    """
    dataset = gdal.Open(in_file_path)

    if dataset is None:
        raise FileNotFoundError(f"Could not open file: {in_file_path}")

    probability: npt.NDArray = dataset.GetRasterBand(1).ReadAsArray()

    print("Shape:", probability.shape)
    print("Min:", np.nanmin(probability))
    print("Max:", np.nanmax(probability))
    print("Number of NaN:", np.isnan(probability).sum())
    print("Total pixels:", probability.size)
    print("NaN fraction:", np.isnan(probability).mean())

    plt.figure(figsize=(8, 8))

    plt.imshow(
        probability,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    plt.title("Probability map")
    plt.colorbar(label="Probability")
    plt.tight_layout()
    plt.show()


def plot_largest_contour(
    raster: npt.NDArray,
    contours: list[npt.NDArray],
) -> None:
    """
    Plot a raster and its largest contour.

    Parameters
    ----------
    raster : npt.NDArray
        Two-dimensional raster to display.
    contours : list[npt.NDArray]
        Contours returned by contourline_segmentation().
    level : float
        Contour level used for the plot title.
    """
    if len(contours) == 0:
        raise RuntimeError("No contour found")

    polygon = max(contours, key=len)

    # find_contours returns (row, column) coordinates.
    polygon_y = polygon[:, 0]
    polygon_x = polygon[:, 1]

    print("Selected contour points:", len(polygon))

    plt.figure(figsize=(10, 10))

    plt.imshow(raster, cmap="gray", vmin=0, vmax=1)

    plt.plot(polygon_x, polygon_y, "-r", linewidth=2)

    plt.colorbar(label="Probability")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()


def plot_contour(
    raster: npt.NDArray,
    contours: list[npt.NDArray],
    contour_index: int,
) -> None:
    """
    Plot a raster and a selected contour.

    Parameters
    ----------
    raster : npt.NDArray
        Two-dimensional raster to display.
    contours : list[npt.NDArray]
        Contours returned by contourline_segmentation().
    contour_index : int
        Index of the contour to plot.
    level : float
        Contour level used for the plot title.

    Raises
    ------
    IndexError
        If contour_index is outside the valid range.
    """
    if not -len(contours) <= contour_index < len(contours):
        raise IndexError(
            f"contour_index must be between {-len(contours)} and {len(contours) - 1}"
        )

    contour = contours[contour_index]

    # find_contours returns (row, column) coordinates.
    polygon_y = contour[:, 0]
    polygon_x = contour[:, 1]

    plt.figure(figsize=(10, 10))

    plt.imshow(raster, cmap="gray", vmin=0, vmax=1)
    plt.plot(polygon_x, polygon_y, "-r", linewidth=2)

    plt.colorbar(label="Probability")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.show()
