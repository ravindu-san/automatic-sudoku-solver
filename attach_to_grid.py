import cv2
import numpy as np
from utils import order_points


def unwrap_image(img_src, img_dst, ordered_points, crop_indices):

    ordered_points = np.array(ordered_points)

    pts_source = crop_indices
    homography, status = cv2.findHomography(pts_source, ordered_points)

    warped = cv2.warpPerspective(img_src, homography, (img_dst.shape[1], img_dst.shape[0]))

    cv2.fillConvexPoly(img_dst, ordered_points, 0, 16)

    # Add the un-warped sudoku puzzle to the black area
    unwrapped_image = cv2.add(img_dst, warped)

    return unwrapped_image


def print_on_screen(puzzle, print_list, solution, img_result, puzzle_contour, crop_indices):

    output_image = puzzle.copy()

    for val in print_list:
        start_x, start_y, end_x, end_y = val['location']

        # The font size as a scale
        font_scale = (end_x - start_x) / 50

        thickness = (end_x - start_x) // 30

        # bottom left corner coordinate to print number
        text_x = int((end_x - start_x) * 0.33)
        text_y = int((end_y - start_y) * -0.2)
        text_x += start_x
        text_y += end_y

        index = val['index']
        cv2.putText(output_image, str(solution[index[1]][index[0]]), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    original_image = img_result.copy()
    wrapped_image = output_image.copy()

    ordered_points = order_points(puzzle_contour.reshape(4, 2)).astype('int32')

    print_image = unwrap_image(wrapped_image, original_image, ordered_points, crop_indices)

    return print_image
