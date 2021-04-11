# DEBUG = True

PUZZLE_LENGTH = 9
BLOCK_SIZE = 3
PUZZLE_SIZE = 81

IDX_EMPTY = -1
IDX_NOT_SET = -2

SOLVED_PUZZLE = None


# def dbg(name, value):
#     if 'DEBUG' in globals() and DEBUG:
#         print(name + ": " + str(value))


def _1d_to_2d(_1_d):
    _1_d = _1_d[:]
    _2_d = []
    for row in range(PUZZLE_LENGTH):
        a_row = []
        for col in range(PUZZLE_LENGTH):
            a_row.append(_1_d.pop(0))
        _2_d.append(a_row)
    return _2_d


def _2d_to_1d(_2_d):
    return [j for sub in _2_d for j in sub]


'''*
 * Get the index for a given coordinate pair in the puzzle vector
 * @param i Vertical position
 * @param j Horizontal position
 * @return Absolute index in the puzzle vector
 '''


def at(i, j):
    return i * PUZZLE_LENGTH + j


'''*
 * Get the coordinate pair for a given index in the puzzle vector
 * @param idx Absolute index in the puzzle vector
 * @return Coordinate pair in the puzzle
 '''


def coord(idx):
    i = idx // PUZZLE_LENGTH
    j = idx % PUZZLE_LENGTH
    return [i, j]


'''*
 * Get the bitmask for value
 * @param value Original number
 * @return Bitmask
 '''


def mask(value):
    return 0 if value == 0 else 1 << (value - 1)


'''*
 * Get the value represented by a bitmask
 * @param mask Bitmask
 * @return Value
 '''


def unmask(mask):
    if mask <= 0:
        return 0

    value = 1
    while (mask):
        mask >>= 1
        if (mask): value += 1
    return value


'''*
 * Count the total number of set bits (1s) in a bitmask
 * @param mask Bitmask
 * @return Count of set bits
 '''


def countBits(mask):
    count = 0
    while (mask):
        mask &= (mask - 1)
        count += 1

    return count


'''*
 * Print a puzzle matrix
 * @param matrix Vector of puzzle values
 * @param title Title
 * @param indent Indent when printing if needed
 '''


def printMatrix(matrix, title=""):
    print(title + ":")
    for i in range(PUZZLE_LENGTH):
        for j in range(PUZZLE_LENGTH):
            print(str(unmask(matrix[at(i, j)])), end=" ")
        print()


'''*
 * Update the permitted values set for a given index
 * @param puzzle Puzzle values vector
 * @param allowed Allowed values vector
 * @param idx Updated index
 '''


def updateAllowed(puzzle, allowed, idx):
    allowed[idx] = 0

    pos = coord(idx)
    i = pos[0]
    j = pos[1]

    safeMask = ~(puzzle[idx])

    ''' disallow value for the block '''
    block_i = i - i % BLOCK_SIZE
    block_j = j - j % BLOCK_SIZE
    for _i in range(BLOCK_SIZE):
        for _j in range(BLOCK_SIZE):
            allowed[at(block_i + _i, block_j + _j)] &= safeMask

    ''' disallow value for the that column & row '''
    for _ in range(PUZZLE_LENGTH):
        allowed[at(_, j)] &= safeMask
        allowed[at(i, _)] &= safeMask


'''*
 * Get an estimated value to represents the variability of neighbours (indices in the same row, or block).
 * This is a heuristic value to get an idea how stable if a value is set to a given index; if the return value is
 * lower mean change of the given index is more stable and priority should be given to fill it first.
 * @param puzzle Puzzle values vector
 * @param allowed Allowed values vector
 * @param idx Index to consider
 * @return Estimated value of variability of neighbours
 '''


def countNeighboursVariability(puzzle, allowed, idx):
    pos = coord(idx)
    i = pos[0]
    j = pos[1]

    count = 0

    block_i = i - i % BLOCK_SIZE
    block_j = j - j % BLOCK_SIZE

    for _i in range(block_i, block_i + BLOCK_SIZE):
        if (_i == i): continue
        for _j in range(block_j, block_j + BLOCK_SIZE):
            if (_j == j): continue
            if (not puzzle[at(_i, _j)]): count += 1

    for _ in range(PUZZLE_LENGTH):
        if (not puzzle[at(_, j)]): count += 1
        if (not puzzle[at(i, _)]): count += 1

    return count + countBits(allowed[idx])


'''*
 * Get the index to be filled next
 * @param puzzle Puzzle values vector
 * @param allowed Allowed values vector
 * @return An index
 '''


def getToFillIndex(puzzle, allowed):
    toFillPrioritiesAndIndices = []

    for i in range(PUZZLE_SIZE):
        if puzzle[i] == 0:
            toFillPrioritiesAndIndices.append([
                countBits(allowed[i]),
                countNeighboursVariability(puzzle, allowed, i),
                i
            ])

    if len(toFillPrioritiesAndIndices) < 1:
        return IDX_EMPTY

    toFillPrioritiesAndIndices.sort(key=lambda x: x[1])
    toFillPrioritiesAndIndices.sort(key=lambda x: x[0])

    # dbg("toFillPrioritiesAndIndices", toFillPrioritiesAndIndices)
    return toFillPrioritiesAndIndices[0][2]


'''*
 * Get a vector of possible values (with a prioritised order) to use to fill a given index.
 * The priority is given by a heuristic estimation based on the previous appearance of the
 * values.
 * @param puzzle Puzzle values vector
 * @param allowed Allowed values vector
 * @param index Index to fill
 * @return Vector of possible values (ordered)
 '''


def getPrioritisedValues(puzzle, allowed, idx):
    appearedCountsMap = {}

    for i in range(PUZZLE_SIZE):
        if (allowed[idx] & puzzle[i]) > 0:
            val = unmask(puzzle[i])
            if not val in appearedCountsMap:
                appearedCountsMap[val] = 1
            else:
                appearedCountsMap[val] += 1

    # dbg("\nappearedCountsMap", appearedCountsMap)
    # dbg("\nappearedCountsMapL", list(dict(sorted(appearedCountsMap.items(), key=lambda item: item[1])).keys()))

    return list(dict(sorted(appearedCountsMap.items(), key=lambda item: item[1])).keys())


'''*
 * Solve the puzzle recursively by brute-forcing. This function tries to fill a single index
 * in a single function call; and call for next index with a value.
 * @param puzzle Puzzle values vector
 * @param allowed Allowed values vector
 * @param idx Index to fill
 * @param value Value to fill to the given index
 * @return True if all empty values are fill, False
 '''


def solvePuzzleRecursive(puzzle, allowed, idx=IDX_NOT_SET, value=0):
    global SOLVED_PUZZLE

    if (idx == IDX_NOT_SET):  # ''' no index is selected to fill (i.e. initial function call) '''
        idx = getToFillIndex(puzzle, allowed)

    if (value > 0):  # ''' set value and update allowed vector '''
        puzzle[idx] = mask(value)
        updateAllowed(puzzle, allowed, idx)
        idx = getToFillIndex(puzzle, allowed);  # ''' more to next index '''

    if (idx == IDX_EMPTY):  # ''' all indices of puzzle are filled '''
        SOLVED_PUZZLE = puzzle
        return True

    prioritisedValues = getPrioritisedValues(puzzle, allowed, idx);
    for val in prioritisedValues:
        if (allowed[idx] & mask(val)) > 0:
            if solvePuzzleRecursive(puzzle[:], allowed[:], idx, val):
                return True

    return False


'''*
 * Solve the puzzle 
 * @param puzzle_input Raw puzzle input as an 1-D array
 * @return solved puzzle as a 1-D array
 '''


def solve_1d(puzzle_input):
    if len(puzzle_input) != PUZZLE_SIZE:
        raise Exception("Expected input size:" + str(PUZZLE_SIZE) + ". But found:" + str(len(puzzle)))

    puzzle = [0] * PUZZLE_SIZE
    allowed = [mask(PUZZLE_LENGTH + 1) - 1] * PUZZLE_SIZE
    for idx in range(PUZZLE_SIZE):
        puzzle[idx] = mask(puzzle_input[idx])

    for idx in range(PUZZLE_SIZE):
        if (puzzle[idx] > 0):
            updateAllowed(puzzle, allowed, idx)

    # printMatrix(puzzle,"Input")

    solved = solvePuzzleRecursive(puzzle, allowed)

    if solved:
        # printMatrix(SOLVED_PUZZLE, "\nOutput")

        solution = []
        for idx in range(PUZZLE_SIZE):
            solution.append(unmask(SOLVED_PUZZLE[idx]))
        return [1, solution]
    else:
        return [0, []]


'''*
 * Wrapper of solve_1d function for 2-D array input
 * @param puzzle_input Raw puzzle input as an 1-D array
 * @return solved puzzle as a 2-D array
 '''


def solve_2d(puzzle_input):
    result = solve_1d(_2d_to_1d(puzzle_input))

    if result[0] != 1:
        return result

    else:
        return [1, _1d_to_2d(result[1])]
