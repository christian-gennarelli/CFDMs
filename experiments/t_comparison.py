import sys
sys.path.insert(0, " thesis") 

import argparse

import numpy as np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt

from algorithms import smoothed_from_T, plain_smoothed
from targets import PointCloud
from w2_distance import w2_dist

def main():

    '''
    Experiment used to measure either of these two:
    1. The distance between the samples distribution obtained with starting timestep T=0 and the one obtained at T=t
    2. The distance between an empirical proxy (used to approximate the ground-truth continuos data distribution) and the samples
       distribution obtained with starting timestep T=t
    This comparison can also be done for several values of sigma.
    '''

    target = PointCloud('bed')
    X = target.sample(500)
    # proxy = target.sample(500)

    T_list = args.T_list
    sigma_list = args.sigmas

    dist = []
    for sigma in sigma_list:
        z_0 = plain_smoothed(
            X = X,
            S = 100,
            K = 1000,
            sigma = sigma,
            M = 10,
        )[:, -1, :]

        dist_T = []

        for t in T_list:
            z = smoothed_from_T(
                X = X,
                S = 100,
                K = 1000,
                sigma = sigma,
                M = 10,
                start_T = t
            )[:, -1, :]

            ###############################################################################
            # W2-Wasserstein distance between proxy and samples obtained with start_T = T # 
            # w2_distance = w2_dist(proxy, z)
            ###############################################################################
            # W2-Wasserstein distance between samples from start_T = 0 and start_T = T #
            w2_distance = w2_dist(z_0, z)
            ###############################################################################
            
            dist_T.append(w2_distance)
        dist.append(dist_T)
    
    plt.xlabel('Starting timestep')
    plt.ylabel('W2 distance')

    for (i, sigma) in enumerate(sigma_list):
        plt.plot(T_list, dist[i], label=f'{sigma}')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    def check_valid_S(sigmas):
        for s in sigmas:
            if float(s) < 0: return False
        return True
    parser.add_argument(
        '-s',
        '--sigmas',
        nargs = '+',
        type = check_valid_S,
        required = True,
        help = 'list of smoothing parameters'
    )

    def check_valid_T(T_list):
        for T in T_list:
            if float(T) < 0 or float(T) > 1: return False
        return True
    parser.add_argument(
        '-T',
        '--T_list',
        nargs = '+',
        type = check_valid_T,
        required = True,
        help = 'list of starting timesteps'
    )

    args = parser.parse_args()

    main(args)