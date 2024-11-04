import sys
sys.path.insert(0, " thesis") 

import argparse

import numpy as np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt

from algorithms import plain_smoothed
from targets import PointCloud
from w2_distance import w2_dist

def main(args):

    '''
    Experiment to compute, for several values of sigma, the distance between an approximation of the data distribution
    (given by the proxy) and the samples obtained with a particular value of the smoothing parameter.
    '''

    sigma_list = args.sigmas
    w2_distances = []
    target = PointCloud('bed')
    X = target.sample(150)
    proxy = target.sample(500)
    print(f'proxy.shape: {proxy.shape}')
    for sigma in sigma_list:
        z = plain_smoothed(
            X = X,
            S = 100,
            K = 1000,
            sigma = sigma,
            M = 10
        )[:, -1, :]

        w2_distances.append(w2_dist(proxy, z))

    # print(distances)

    plt.xlabel('Smoothing parameter')
    plt.ylabel('W2-Wasserstein distance')
    plt.plot(sigma_list, w2_distances, 'red')

    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    def check_valid_s(sigmas):
        for s in sigmas:
            if float(s) < 0: return False
        return True
    parser.add_argument(
        '-s',
        '--sigmas',
        nargs = '+',
        type = check_valid_s,
        required = True,
        help = 'list of smoothing parameters'
    )

    args = parser.parse_args()

    main(args)