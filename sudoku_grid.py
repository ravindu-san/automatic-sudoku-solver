import numpy as np
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from utils import order_points

MIN_CORNER_DIST = 50


def check_if_puzzle(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    temp = c.copy().reshape(4, 2)

    # Take one corner
    p1 = temp[0]

    count_x = 0
    count_y = 0

    # Check if any length from the selected corner and other corners is less than 50
    for i in range(1, len(temp)):

        if abs(temp[i][0] - p1[0]) < MIN_CORNER_DIST:
            count_x += 1

        if abs(temp[i][1] - p1[1]) < MIN_CORNER_DIST:
            count_y += 1

    if count_x > 1 or count_y > 1:
        return False

    if len(approx) == 4:
        return True

    return False


def four_point_transform(image, pts):
    rect = order_points(pts)

    # Top-left, top-right, bottom-right, bottom-left coordinates
    (tl, tr, br, bl) = rect

    # Find width and heights
    width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # max of top width and bottom width to make sure no important part is cropped out
    max_width = max(int(width_1), int(width_2))

    height_1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_1), int(height_2))

    crop_indices = np.array([
        [0, 0],  # Index for upper left corner
        [max_width - 1, 0],  # Index for upper right corner (-1 because it is an index)
        [max_width - 1, max_height - 1],  # Index for lower right corner
        [0, max_height - 1]], dtype="float32")  # Index for lower left corner

    # Take the corner coordinates and warp them to fit the index above
    perspective_trans = cv2.getPerspectiveTransform(rect, crop_indices)
    warped = cv2.warpPerspective(image, perspective_trans, (max_width, max_height))

    return warped, crop_indices


def extract_grid(image, debug=False):
    # rgb image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # threshold
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colors
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow(" ", thresh)
        cv2.waitKey(0)

    # contours
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sort contours by area
    # descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    puzzle_contour = None

    for c in contours:
        # contour perimeter
        peri = cv2.arcLength(c, True)

        # Find an approximate shape
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # first 4 sided contour is the puzzle box.
        if len(approx) == 4:
            puzzle_contour = approx
            break

    if puzzle_contour is None:
        return None

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzle_contour], -1, (0, 255, 0), 2)
        cv2.imshow(" ", output)
        cv2.waitKey(0)

    # The warped extracted puzzle and its corner coordinates (indexes)
    puzzle_wp_colour, crop_indices = four_point_transform(image, puzzle_contour.reshape(4, 2))

    # The warped extracted puzzle (grayscale) and its corner coordinates (indexes)
    puzzle_wp_gray, crop_indices = four_point_transform(gray, puzzle_contour.reshape(4, 2))

    if debug:
        cv2.imshow(" ", puzzle_wp_colour)
    return puzzle_wp_colour, puzzle_wp_gray, puzzle_contour, crop_indices


def extract_digit(cell, debug=False):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    # Get the contour with the maximum areas
    digit_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Take black image, draw the max contour. 1st arg: black image, 2nd: list of contours (only one here)
    # 3rd: index of contour (-1 means draw all in the list), 4th: Colour 5th: Thickness (-1 means fill it)
    cv2.drawContours(mask, [digit_contour], -1, 255, -1)

    (h, w) = thresh.shape
    # See which percentage of image is filled by the above filled contour
    percent_filled = cv2.countNonZero(mask) / float(w * h)

    if percent_filled < 0.03:
        return None

    # Using the above mask using the and operation extract the digit from the image input
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    if debug:
        cv2.imshow(" ", digit)
        cv2.waitKey(0)

    return digit


def check_allowed_values(allowed_values, x, y, value, box_dim=3):
    # Remove value from allowed list in row with the value
    for idy, allow in enumerate(allowed_values[x]):
        allowed_values[x][idy][value] = False

    # Remove value from allowed list in column with the value
    for idx, allow in enumerate(allowed_values):
        allowed_values[idx][y][value] = False

    begin_row = x - x % box_dim
    begin_column = y - y % box_dim

    # Remove value from allowed list in block with the value
    for i in range(begin_row, begin_row + box_dim):
        for j in range(begin_column, begin_column + box_dim):
            allowed_values[i][j][value] = False


def get_the_grid(img_result, model):
    # The warped puzzle,warped puzzle (grayscale), puzzle contour coordinates (in image), warped puzzle corner indices
    # found_puzzle = find_puzzle(img_result, False)
    found_puzzle = extract_grid(img_result, False)

    if found_puzzle is None:
        return None
    else:
        puzzle_wp_colour, puzzle_wp_gray, puzzle_contour, dst = found_puzzle

    # If puzzle contour was not found
    if puzzle_contour is None:
        return None

    # Check if puzzle qualifies for sudoku puzzle (the size of the puzzle)
    if not check_if_puzzle(puzzle_contour):
        return None

    # four_point_transform(img_result.copy(), puzzle_contour.reshape(4, 2))

    board = np.zeros((9, 9), dtype="int")
    print_list = []
    digit_count = 0

    allowed_values = [[[True for _ in range(9)] for _ in range(9)]
                      for _ in range(9)]

    # The length of a cell
    step_x = puzzle_wp_gray.shape[1] // 9
    step_y = puzzle_wp_gray.shape[0] // 9

    # Loop over the grid locations
    for y in range(0, 9):
        for x in range(0, 9):

            # Compute the starting and ending (x, y)-coordinates of the current cell
            start_x = x * step_x
            start_y = y * step_y
            end_x = (x + 1) * step_x
            end_y = (y + 1) * step_y

            # Crop the cell from the warped transform image and then extract the digit from the cell
            cell = puzzle_wp_gray[start_y:end_y, start_x:end_x]
            digit = extract_digit(cell, False)

            # Verify that the cell is not empty
            if digit is not None:

                # Resize the cell to 28x28
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Classify the digit and update the Sudoku board with the prediction
                prediction = model.predict(roi)[0]
                predict_num = np.argsort(-prediction)[:3]
                added_num = False
                for value in predict_num:
                    if value > 0:
                        if allowed_values[y][x][value - 1]:
                            check_allowed_values(allowed_values, y, x, value - 1)
                            board[y, x] = value
                            digit_count += 1
                            added_num = True
                            break
                if not added_num:
                    return None
            else:

                # Coordinates of image to print if it is an empty cell
                print_list.append({"index": (x, y), "location": (start_x, start_y, end_x, end_y)})

    if digit_count < 17:
        return None

    # print(board)
    return puzzle_wp_colour, puzzle_contour, dst, board, print_list
