import sys
sys.path.insert(0, " thesis") 

import argparse
from random import randint

import numpy as np
import matplotlib.pyplot as plt

from algorithms import sampling_loop
from scores import SmoothedScore
from targets import Checkerboard, Spirals, Moons, SwissRoll

def sample(X, sigmas, S, M):
    z_sigmas = None
    for sigma in sigmas:
        # print('sigma:', sigma, '\n')
        z = sampling_loop(
            SmoothedScore(
                X = X,
                sigma = sigma,
                noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M)
            ),
            S = S,
            z = None,
            start_T = 0,
            final_T = 1
        )
        z_sigmas = np.expand_dims(np.array(z), axis=0) if z_sigmas is None else np.concatenate((z_sigmas, np.expand_dims(z, axis=0)))
    return z_sigmas

def scatterplot_varying_sigma(X, z_list, sigmas):

    colors = ['#%06X' % randint(0, 0xFFFFFF) for _ in range(len(sigmas))]

    fig, axs = plt.subplots(
        nrows=max(len(sigmas)//3, 1),  
        ncols=min(len(sigmas), 3), 
        sharex=False, 
        sharey=False
    )

    for i, sigma in enumerate(sigmas):
        ax = axs.flat[i]
        ax.scatter(X[:,0], X[:,1], c='red', label='Target distribution')
        ax.scatter(z_list[i,:,0], z_list[i,:,1], c=colors[i], label='Sampled points', alpha=0.5)
        ax.set_title(f'Sigma: %.2f' %sigma)

    # Remove extra empty plots    
    for i in range(len(sigmas), len(axs.flat)):
        fig.delaxes(axs.flat[i])

    # fig.suptitle(title)
    # fig.text(.5, .05, caption, fontsize='small', ha='center', wrap=True)
    plt.show()

def main(args):

    '''
    Experiment allowing to plot different set of samples against 
    their training set for several values of the smoothing parameter.
    '''

    match args.target:
        case 'swissroll': target = SwissRoll()
        case 'checkerboard': target = Checkerboard()
        case 'spirals': target = Spirals()
        case 'moons': target = Moons()

    X = target.sample(500)
    sigmas = np.array(args.sigmas)
    S = 100
    M = 2

    z_list = None
    K = 500
    for _ in range(K):
        z = sample(X, sigmas, S, M)
        z_list = np.expand_dims(np.array(z), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(z, axis=0)), axis=0)

    scatterplot_varying_sigma(X, np.swapaxes(z_list[:, :, -1, :], 0, 1), sigmas)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--target',
        type = str,
        choices = ['swissroll', 'checkerboard', 'spirals', 'moons'],
        required = True,
        help = 'target distribution'
    )
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