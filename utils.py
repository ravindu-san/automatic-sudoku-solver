import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # top-left point -> smallest sum
    # bottom-right point -> largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute difference between the points
    # top-right point -> smallest difference,
    # the bottom-left -> largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
