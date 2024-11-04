import sys
sys.path.insert(0, " thesis") 

import argparse
from time import time

import numpy as np
import matplotlib.pyplot as plt

from algorithms import smoothedDEANN, plain_smoothed
from targets import PointCloud

def main(target_names):

    '''
    Experiment showing the differences in terms of running times between smoothed and smoothedDEANN,
    for reconstructing partially corrupted point clouds.
    '''

    K = 5000
    S = 100
    n = 150
    sigma = 0.08
    M = 10

    rt_total = []
    for target_name in target_names:
        target = PointCloud(target_name)
        X = target.sample(n)

        rt = []

        start = time()
        _ = plain_smoothed(
            X,
            S, 
            K, 
            sigma, 
            M,
        )
        end = time()
        print(end-start)
        rt.append(end - start)

        start = time()
        _ = smoothedDEANN(
            X,
            S,
            K, 
            sigma,
            M,
            15, 
            5, 
            15,
        )
        end = time()
        print(end - start)
        rt.append(end-start)

        rt_total.append(rt)
    
    rt_total = np.array(rt_total)
    X_axis = np.arange(len(target_names))
    plt.bar(X_axis - 0.2, rt_total[:, 0], 0.4, label = 'smoothed_from_T')
    plt.bar(X_axis + 0.2, rt_total[:, 1], 0.4, label = 'smoothedDEANN_from_T')
    plt.xticks(X_axis, target_names) 
    plt.ylabel("Runtime (in seconds)") 
    plt.legend() 
    plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--targets',
        nargs = '+',
        type = 'str',
        required = True,
        help = 'list of targets to be used',
    )
    args = parser.parse_args()

    main(args.targets)