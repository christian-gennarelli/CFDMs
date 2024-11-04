import sys
sys.path.insert(0, "/Users/genna/Documents/ACSAI/Closed-Form DM/thesis") 

from algorithms import smoothed_from_T
from targets import PointCloud
from w2_distance import w2_dist

import numpy as np
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
import polyscope as ps; ps.init()

from time import time

def main():
    
    N_list = [100, 500, 1000, 2000, 3000, 4000, 5000]

    target = PointCloud('bed')
    proxy = target.sample(5000)
    print('Proxy sampled.')

    w2_distances = []
    runtimes = []
    for N in N_list:

        X = target.sample(N)
        # ps.register_point_cloud(
        #     name = f'train_{N}',
        #     points = X
        # )

        # start = time()
        z = smoothed_from_T(
            X = X,
            S = 100,
            K = 15000,
            sigma = 0.08,
            M = 10,
            start_T = 0.95
        )[:, -1, :]
        ps.register_point_cloud(
            name = f'cloud_{N}',
            points = z
        )
        # end = time()
    
    ps.show()
        
    #     runtimes.append(end - start)
    #     w2_distances.append(w2_dist(proxy, z))
    
    # plt.xticks(N_list, N_list)
    # plt.ylabel('W2 distance')
    # plt.xlabel('Number of training points')
    # plt.plot(N_list, w2_distances)
    # plt.show()

    # plt.xticks(N_list, N_list)
    # plt.ylabel('Runtime (in seconds)')
    # plt.xlabel('Number of training points')
    # plt.plot(N_list, runtimes)
    # plt.show()


if __name__ == '__main__':
    main()