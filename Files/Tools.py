import numpy as np
import math

def gridFunction(grid_min, grid_max, num_grids):

    assert grid_max > grid_min and num_grids >= 2

    grid = np.zeros(num_grids)
    grid[0] = grid_min
    for i in range(1, num_grids):
        numerator = grid_max - grid[i - 1]
        denominator = (num_grids - i) ** 1.1
        grid[i] = numerator/denominator + grid[i - 1]

    assert np.all(np.diff(grid)>0) # Made for cowards

    return grid

def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def GaussHermite_lognorm(sigma,n):

    x, w = gauss_hermite(n)
    x = np.exp(x*math.sqrt(2)*sigma - 0.5*sigma**2)
    w = w / math.sqrt(math.pi)

    # assert a mean of one
    assert(1 - np.sum(w*x) < 1e-8 ), 'The mean in GH-lognorm is not 1'
    return x, w