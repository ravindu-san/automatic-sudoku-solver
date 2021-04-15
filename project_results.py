import cv2
import numpy as np
from utils import order_points


def unwrap_image(img_src, img_dst, ordered_points, crop_indices):

    ordered_points = np.array(ordered_points)
    source_pts = crop_indices

    # register four point transformed sudoku grid with original sudoku grid
    homography, status = cv2.findHomography(source_pts , ordered_points)
    warped = cv2.warpPerspective(img_src, homography, (img_dst.shape[1], img_dst.shape[0]))

    # insert sudoku grid with results to the area of original sudoku grid in original image
    cv2.fillConvexPoly(img_dst, ordered_points, 0, 16)
    unwrapped_image = cv2.add(img_dst, warped)

    # cv2.imwrite("./output_images/result_project.jpg", unwrapped_image)

    return unwrapped_image


def project_to_original_img(puzzle, print_list, solution, img_result, puzzle_contour, crop_indices):

    output_img = puzzle.copy()

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
        cv2.putText(output_img, str(solution[index[1]][index[0]]), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    original_img = img_result.copy()
    wrapped_img = output_img.copy()

    ordered_points = order_points(puzzle_contour.reshape(4, 2)).astype('int32')

    print_img = unwrap_image(wrapped_img, original_img, ordered_points, crop_indices)

    return print_img
