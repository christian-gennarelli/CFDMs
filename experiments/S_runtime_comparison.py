import sys
sys.path.insert(0, " thesis") 

import argparse
from time import time

import matplotlib.pyplot as plt

from algorithms import plain_smoothed
from targets import PointCloud

def main(args):

    '''
    Experiment showing the (linear growing) runtime of CFDMs when using an increasing number of steps 
    to better approximate the velocity field.
    '''

    X = PointCloud('bed').sample(150)

    S_list = args.S
    runtimes = []

    for S in S_list:

        start = time()
        plain_smoothed(
            X = X, 
            S = S, 
            K = 5000,
            sigma = 0.08,
            M = 10
        )
        end = time()
        runtime = end - start
        print(f'Runtime for S={S}: {runtime}')
        runtimes.append(runtime)
    
    plt.xticks(S_list, S_list)
    plt.xlabel('Number of steps')
    plt.ylabel('Runtime (in seconds)')
    plt.plot(S_list, runtimes)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    def check_valid_S(S_list):
        for S in S_list:
            if int(S) <= 0: return False
        return True
    parser.add_argument(
        '-S',
        '--S_list',
        nargs = '+',
        type = check_valid_S,
        required = True,
        help = 'list of number of steps to perform'
    )
    
    args = parser.parse_args()

    main(args)