import numpy as np
def gridmat(N, shape = None, order = True, seed = None):
    """
    Spatial coordinate matrix in 2-D space, x and y axis are equally spacing.

    Parameter
    ---------
    N : int        
        Number of neurons (nodes)
    shape : tuple of int
        shape of 2-D grid matrix
        (default : None. Return a smallest square
        grid)
    order : bool
        Nodes are orderly embeded in grid or not
        True for ordered, False for randomized

    Return
    ------
    xmat : 2-D array of int
        matrix of x normalized coordinate
    ymat : 2-D array of int
        matrix of y normalized coordinate

    """
    if shape is None:
        gd_size = int(np.sqrt(N))
        X,Y = np.meshgrid(np.arange(gd_size, dtype=float),np.arange(gd_size, dtype=float))
        X = X / gd_size
        Y = Y / gd_size
        if order:
            return X, Y
        else:
            if seed is not None:
                np.random.seed(seed)
            perm_id = np.random.permutation(gd_size*gd_size)
            X = X.flatten()[perm_id].reshape((gd_size, gd_size))
            Y = Y.flatten()[perm_id].reshape((gd_size, gd_size))
            return X, Y
    else:
        if shape[0]*shape[1] >= N:
            X,Y = np.meshgrid(np.arange(shape[0], dtype=float),np.arange(shape[1], dtype=float))
            X = X / max(shape)
            Y = Y / max(shape)
            if order:
                return X, Y
            else:
                if seed is not None:
                    np.random.seed(seed)
                perm_id = np.random.permutation(shape[0]*shape[1])
                X = X.flatten()[perm_id].reshape(shape)
                Y = Y.flatten()[perm_id].reshape(shape)
                return X, Y
        else:
            print("ERROR : (%d,%d) grid cannot containing %d neurons (nodes)." % (shape[0], shape[1], N))

def gen_delay_matrix(grid, weight):
    from scipy.spatial.distance import cdist
    return weight * cdist(grid, grid, metric='euclidean')
