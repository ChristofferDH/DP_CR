import numpy as np

def gridFunction(grid_min, grid_max, num_grids):

    assert grid_max > grid_min
    assert num_grids >= 2

    grid = np.zeros(num_grids)
    grid[0] = grid_min
    for i in range(1, num_grids):
        numerator = grid_max - grid[i - 1]
        denominator = (num_grids - i) ** 1.1
        grid[i] = numerator/denominator + grid[i - 1]

    assert np.all(np.diff(grid)>0) # Made for cowards

    return grid