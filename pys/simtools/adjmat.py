import numpy as np
def randnet(shape, p, seed = None, auto_recurrent = False):
    """
    Generate adjacent matrix for n-neuron network
    
    Parameters
    ==========
    shape : 2-D tuple of intergers or single interger
        shape of adjacent matrix, following the conventional pre-and-post indexing rule
        or dimension of square matrix

    p : float
        pecentage of efferent connections of each neuron

    seed : int
        seed for random generator, default None

    auto_recurrent : bool
        whether allow self recurrent connection

    Return
    ======
    mat : m-by-n ndarray of intergers

    """
    if seed != None:
        np.random.seed(seed)
    if type(shape) == tuple:
        n_con = int(shape[0]*p)
        mat = np.zeros(shape)
        for i in range(shape[1]):
            if auto_recurrent:
                mat[np.random.choice(np.arange(shape[0]), n_con, replace=False), i] = 1
            else:
                if i >= shape[0]:
                    mat[np.random.choice(np.arange(shape[0]), n_con, replace=False), i] = 1
                else:
                    mat[np.random.choice(np.delete(np.arange(shape[0]), i), n_con, replace=False), i] = 1
        return mat    
    elif type(shape) == int:
        n_con = int(shape*p)
        mat = np.zeros((shape, shape))
        for i in range(shape):
            if auto_recurrent:
                mat[np.random.choice(np.arange(shape), n_con, replace=False), i] = 1
            else:
                mat[np.random.choice(np.delete(np.arange(shape), i), n_con, replace=False), i] = 1
        return mat    
    else:
        print("ERROR : invalid shape")

def regularnet(shape, k):
    """
    Generate adjacent matrix for n-neuron regular ring network
    
    Parameters
    ==========
    shape : interger
        dimension of square matrix

    k : float
        half number of neighborhood of each neuron,
        i.e., each neuron connects with 2*k neighoring neuron

    Return
    ======
    mat : n-by-n ndarray of intergers

    """
    mat = np.zeros((shape,shape))
    for i in range(int(k)):
        mat += np.eye(shape, k= i+1)
        mat += np.eye(shape, k=-i-1)
        mat += np.eye(shape, k= shape-1-i)
        mat += np.eye(shape, k= -shape+1+i)
    return mat

# Spatially dependent connectivity
# case 1: Gaussian distributed connectivity
def gaussiannet(X, p, sigma_x = 0.1, sigma_y=None, seed = None, directed = True):
    """
    Generate Gaussian distributed connectivity matrix ( Complexity : O(N**2) )
    Connecting probability is proportional to the spatial distance between two nodes

    Parameters
    ==========
    X : N-by-2 array
        coordinate the nodes in 2-D space

    p : float
        Mean connectiong probability

    sigma : float
        Standard deviation of Gaussian distribution

    seed : int
        seed for random generator, default None

    directed : bool
        Directed network or not, default True

    Return
    ======
    mat : N-by-N array of int
        connectivity matrix of network

    """
    N = X.shape[0]
    thred_mat = np.zeros((N,N))
    # calculate distance matrix:
    if sigma_y == None:
        sigma_y = sigma_x
    factor = 0.5*p/np.pi/sigma_x/sigma_y
    for i in range(N):
        thred_mat[i,:] = np.exp(-((X[i,0]-X[:,0])**2/(2*sigma_x**2)+(X[i,1]-X[:,1])**2/(2*sigma_y**2)))

    thred_mat = thred_mat * factor

    if seed != None:
        np.random.seed(seed)
    mat = (np.random.random((N,N))<=thred_mat).astype(int)
    mat -= np.eye(N).astype(int)
    if not directed:
        mat = np.triu(mat, k=1)
        mat += mat.T
    return mat


