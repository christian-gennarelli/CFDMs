import sys
sys.path.insert(0, " thesis") 

import argparse

import numpy as np
import matplotlib.pyplot as plt

from algorithms import plain_smoothed, smoothedDEANN
from targets import Checkerboard, PointCloud

def load_3lines(num_samples):

    y_0 = np.zeros(num_samples)
    y_1 = y_0 + 1.5 
    y_2 = y_1 + 0.5 

    x = np.tile(np.arange(0, num_samples, 1) / num_samples, 3)
    y = np.concatenate((y_0, y_1, y_2), axis=0)

    return np.stack((x, y), axis=1)

def main():

    '''
    Experiment to show the existent tradeoff between the smoothing parameter and the number of noises
    in terms of quality of novel samples with respect to the target distribution.
    '''

    X = load_3lines(50)

    sigma_list = args.sigmas
    M_list = args.noises
    for sigma in sigma_list:
        for M in M_list:
            z_list = plain_smoothed(
                X = X,
                S = 100,
                K = 500,
                sigma = sigma,
                M = M,
            )[:, -1, :]

            caption = f'sigma = {sigma}; M = {M}'

            fig, ax = plt.subplots()
            fig.text(.5, .05, caption, fontsize='small', ha='center', wrap=True)
            ax.scatter(z_list[:,0], z_list[:,1], color='blue')
            ax.scatter(X[:,0], X[:, 1], color='red', alpha=0.5)
            # plt.savefig(f'results/experiments/sigma_noises_tradeoff/sigma{str(sigma).replace(".", "")}_M{M}.png')
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

    def check_valid_M(M_list):
        for M in M_list: 
            if type(M) is not int or M < 0:
                return False
        return True
    parser.add_argument(
        '-m',
        '--noises',
        nargs = '+',
        type = check_valid_M,
        required = True,
        help = 'list of number of noises'
    )

    args = parser.parse_args()

    main(args)