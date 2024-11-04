import sys
sys.path.insert(0, "/Users/genna/Documents/ACSAI/Closed-Form DM/thesis") 

from scores import SmoothedScore

import numpy as np
from scipy.spatial import KDTree

def retrieve_local_density(X, P, r):
    
    tree = KDTree(X)
    idx = tree.query_ball_point(P, r)

    return idx.shape[0] / (4/3 * np.pi * r**3)

def sampling_loop(s, z, S, start_T, final_T):

    h = 1/S
    # Initial sample i.e. z_0 ~ N(0, I)
    if z is None: z = np.random.multivariate_normal(
        mean = np.zeros(s.X.shape[1]),
        cov = np.eye(s.X.shape[1]) * np.std(s.X, axis=0)
    ).reshape(1, -1)

    z_list = None 
    for n in range(int(start_T*S), int(final_T*S)): # n = 0,...,S-1
        t = n*h
        if t != 0:
            
            ### Update smoothing parameter based on local density ###
            

            res = s.forward(t,z)
            v = (1/t)*(z+(1-t)*res)
            z += h*v
        z_list = np.array(z) if z_list is None else np.vstack((z_list, z))
    return z_list

def main():
    pass

if __name__ == '__main__':
    main()