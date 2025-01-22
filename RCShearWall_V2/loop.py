import numpy as np

# Define the matrix
matrix = np.array([[13, 14, 15, 16],
                   [9, 10, 11, 12],
                   [5, 6, 7, 8],
                   [1, 2, 3, 4]])

print(matrix)
# Function to loop through the matrix in the desired order
def zigzag_traversal(mat):
    rows, cols = mat.shape
    result = []

    # Traverse in a zigzag pattern
    for col in range(cols):
        if col % 2 == 0:
            # Even columns: top to bottom
            for row in range(rows):
                result.append(mat[row, col])
        else:
            # Odd columns: bottom to top
            for row in range(rows - 1, -1, -1):
                result.append(mat[row, col])

    return result


# Get the result
result = zigzag_traversal(matrix)

# Print the result
print(result)
