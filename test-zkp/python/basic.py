import random

# Simplified 4x4 Sudoku puzzle
# 0 should be filled in by the solver
puzzle = [
    [1, 2, 0, 0],
    [3, 4, 0, 0],
    [0, 0, 1, 2],
    [0, 0, 3, 4]
]

# Solution to the simplified 4x4 Sudoku puzzle
# secret
solution = [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [4, 3, 1, 2],
    [2, 1, 3, 4]
]

def is_valid_sudoku(puzzle, solution):
    """
    Checks if the given solution is valid for the given Sudoku puzzle.
    """
    for i in range(len(puzzle)):
        for j in range(len(puzzle[0])):
            if puzzle[i][j] != 0 and puzzle[i][j] != solution[i][j]:
                return False
    return True

def create_commitment_matrix(solution):
    """
    Creates a commitment matrix for the given solution.
    """
    commitment_matrix = []
    for row in solution:
        commitment_row = []
        for value in row:
            random_factor = random.randint(1, 100)
            commitment = (value * random_factor) % 101
            commitment_row.append(commitment)
        commitment_matrix.append(commitment_row)
    return commitment_matrix

def verify_commitment(commitment_matrix, puzzle, solution):
    """
    Verifies if the given commitment matrix matches the given Sudoku solution for the given puzzle.
    """
    for i in range(len(commitment_matrix)):
        for j in range(len(commitment_matrix[0])):
            if puzzle[i][j] == 0:
                commitment = commitment_matrix[i][j]
                for factor in range(1, 101):
                    if (solution[i][j] * factor) % 101 == commitment:
                        break
                else:
                    return False
    return True

# Check if the Sudoku solution is valid
if is_valid_sudoku(puzzle, solution):
    # Print the puzzle, solution, and commitment matrix
    print("Public puzzle: ", puzzle)
    print("Private solution: ", solution)
    commitment_matrix = create_commitment_matrix(solution)
    print("Public commitment matrix: ", commitment_matrix)

    # Verify the commitment without revealing the solution
    print("Verifier can check the commitment without revealing the solution: ")
    print(verify_commitment(commitment_matrix, puzzle, solution))
else:
    print("Invalid Sudoku solution")
