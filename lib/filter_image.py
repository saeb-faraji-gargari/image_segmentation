import numpy as np
from matplotlib.path import Path


def filter_nested_and_small_contours(
    contours: list[np.ndarray],
    min_area: float = 100,
) -> list[np.ndarray]:
    """
    Filter contours based on minimum enclosed area and spatial containment.

    The function applies two filtering criteria:

    1. Minimum area filtering:
       Removes contours whose enclosed polygon area is smaller than
       min_area.

    2. Nested contour filtering:
       Sorts the remaining contours by area in descending order.
       Starting with the largest contour, checks whether each subsequent
       contour is completely contained within any previously selected
       contour. Contained contours are discarded, while independent
       contours are retained.

    Parameters
    ----------
    contours : list[np.ndarray]
        List of contour arrays (N, 2), with row and column coordinates.
    min_area : float
        Minimum enclosed contour area in pixels.

    Returns
    -------
    list[np.ndarray]
        Selected contours, sorted from largest to smallest.
    """

    def polygon_area(contour):

        if len(contour) < 3:
            return 0.0

        row = contour[:, 0]
        col = contour[:, 1]

        return 0.5 * abs(np.sum(col * np.roll(row, -1) - row * np.roll(col, -1)))

    # Calculate contour areas
    contour_areas = [polygon_area(contour) for contour in contours]

    # Filter contours based on minimum area
    valid_contours = [
        contour
        for contour, area in zip(contours, contour_areas, strict=True)
        if area >= min_area
    ]

    # Sort from largest to smallest
    valid_contours.sort(
        key=polygon_area,
        reverse=True,
    )

    selected_contours = []

    for contour in valid_contours:
        is_inside = False

        for selected in selected_contours:
            selected_path = Path(selected)

            if np.all(selected_path.contains_points(contour)):
                is_inside = True
                break

        if not is_inside:
            selected_contours.append(contour)

    return selected_contours
