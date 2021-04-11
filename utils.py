import numpy as np
from scipy.spatial import distance as dist


def order_points(pts):
    # four coordinates are sorted based on the x coordinate
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # The leftmost 2 coordinates
    left_most = x_sorted[:2, :]
    # The rightmost 2 coordinates
    right_most = x_sorted[2:, :]

    # top left & bottom left coordinates
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # diagonal corner of top left
    # Highest euclidean distance from tl corner.
    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]

    # Bottom right & top right corners
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")
