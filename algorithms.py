from time import time
from itertools import product

import numpy as np
from scores import *
from plot import *
from targets import *


def plain_unsmoothed(X, S, K):

    '''
    Flow a random initial sample through a velocity field employing the unsmoothed score, 
    using n samples from the target distribution to evaluate it in closed-form.
    '''

    z_list = None
    score = UnsmoothedScore(
        X = X,
    )

    for _ in range(K):
        z = sampling_loop(
            s = score,
            S = S, 
            z = None, 
            start_T = 0, 
            final_T = 1
        )
        z_list = np.expand_dims(np.array(z), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(z, axis=0)))
    return z_list

def plain_smoothed(X, S, K, sigma, M):

    '''
    Flow a random initial sample through a velocity field employing the smoothed score,
    using n samples from the target distribution and M levels of noise to evaluate it in closed-form.
    '''

    z_list = None
    score = SmoothedScore(
        X = X,
        sigma = sigma,
        noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M) 
    )

    for _ in range(K):
        z = sampling_loop(
            s = score,
            S = S, 
            z = None, 
            start_T = 0, 
            final_T = 1
        )
        score.noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M) 
        z_list = np.expand_dims(np.array(z), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(z, axis=0)))
    return z_list

def smoothed_from_T(X, S, K, sigma, M, start_T):

    '''
    At each timestep, the target distribution is simply a mixture of Gaussians centered at training points.
    We exploit this fact to sample a σ-CFDM in fewer steps, by sampling a base sample from such a mixture at time T
    and then flowing it through the velocity field employing the smoothed score up to T=1.
    '''

    z_list = None
    score = SmoothedScore(
        X = X,
        sigma = sigma,
        noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M) 
    )

    for _ in range(K):

        tX = start_T * X
        # Uniformy sample the mixture mean t*x_i
        random_idx = np.random.randint(0, X.shape[0], 1)
        tx = (tX[random_idx, :]).squeeze()
        # Sample from a normal distribution centered in tx and with variance (1-t)^2
        z_unsmoothed = np.random.multivariate_normal(
            mean=tx,
            cov=np.eye(X.shape[1])*((1-start_T)**2)
        ).reshape(1, -1)

        z = sampling_loop(
            s = score,
            S = S, 
            z = z_unsmoothed, 
            start_T = start_T, 
            final_T = 1
        )
        # print(f'Point {_} sampled')
        score.noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M) 
        z_list = np.expand_dims(np.array(np.concatenate((z_unsmoothed, z), axis=0)), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(np.concatenate((z_unsmoothed, z), axis=0), axis=0)))
    return z_list

def unsmoothedDEANN(X, S, K, k, nprobe, l):

    '''
    The score of the perturbed target distribution is also the score of a Gaussian KDE.
    We leverage this obeservation by using DEANN to approximate the sum over N samples
    with just K nearest neighbours and L random samples from the remainder of the dataset.
    Here it is the unsmoothed version, which just returns training samples.
    '''

    z_list = None
    score = UnsmoothedDEANN(
        X = X,
        k = k, 
        l = l,
        nprobe = nprobe
    )

    for _ in range(K):
        z = sampling_loop(
            s = score,
            S = S, 
            z = None, 
            start_T = 0, 
            final_T = 1
        )
        z_list = np.expand_dims(np.array(z), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(z, axis=0)))
    return z_list

def smoothedDEANN(X, S, K, sigma, M, k, nprobe, l):

    '''
    The score of the perturbed target distribution is also the score of a Gaussian KDE.
    We leverage this obeservation by using DEANN to approximate the sum over N samples
    with just K nearest neighbours and L random samples from the remainder of the dataset.
    Here it is the smoothed version, which promotes generalization.
    '''

    z_list = None
    score = SmoothedDEANN(
        X = X,
        sigma = sigma,
        noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M),
        k = k, 
        nprobe = nprobe,
        l = l
    )

    for _ in range(K):

        z = sampling_loop(
            s = score,
            S = S, 
            z = None, 
            start_T = 0, 
            final_T = 1
        )
        score.noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M) 
        z_list = np.expand_dims(np.array(z), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(z, axis=0)), axis=0)
    return z_list

def smoothedDEANN_from_T(X, S, K, sigma, M, k, nprobe, l, start_T):

    z_list = None
    score = SmoothedDEANN(
        X = X,
        sigma = sigma,
        noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M),
        k = k,
        l = l,
        nprobe = nprobe
    )

    for _ in range(K):

        tX = start_T * X
        # Uniformy sample the mixture mean t*x_i
        random_idx = np.random.randint(0, X.shape[0], 1)
        tx = (tX[random_idx, :]).squeeze()
        # Sample from a normal distribution centered in tx and with variance (1-t)^2
        z_unsmoothed = np.random.multivariate_normal(
            mean=tx,
            cov=np.eye(X.shape[1])*((1-start_T)**2)
        ).reshape(1, -1)

        z = sampling_loop(
            s = score,
            S = S, 
            z = z_unsmoothed, 
            start_T = start_T, 
            final_T = 1
        )
        # print(f'Point {_} sampled.')
        score.noise = np.random.multivariate_normal(np.zeros(X.shape[1]), np.eye(X.shape[1]), M) 
        z_list = np.expand_dims(np.array(np.concatenate((z_unsmoothed, z), axis=0)), axis=0) if z_list is None else np.concatenate((z_list, np.expand_dims(np.concatenate((z_unsmoothed, z), axis=0), axis=0)))
    return z_list

def sampling_loop(s, z, S, start_T, final_T):
    
    '''
    Implementation of sampling algorithm (7)

    Inputs:
    - s: score instance
    - z: base sample
    - start_T: initial timestep from which z should be flown
    - final_T: final timestep to which z should be flown

    Outputs:
    - z_list: list of points representing the flow trajectory of z
    '''

    h = 1/S
    # Initial sample i.e. z_0 ~ N(0, I)
    if z is None: z = np.random.multivariate_normal(
        mean = np.zeros(s.X.shape[1]),
        cov = np.eye(s.X.shape[1]) * np.std(s.X, axis=0)
    ).reshape(1, -1)
    # print('Initial z:\n', z, '\n')

    # Trajectory of the generated sample
    z_list = None 
    for n in range(int(start_T*S), int(final_T*S)): # n = 0,...,S-1
        t = n*h
        if t != 0:
            res = s.forward(t,z)
            v = (1/t)*(z+(1-t)*res)
            z += h*v
        z_list = np.array(z) if z_list is None else np.vstack((z_list, z))
    return z_list

def main(args):

    match args.target:
        case 'checkerboard': target = Checkerboard()
        case 'spirals': target = Spirals()
        case 'line': target = Line()
        case 'std_normal': target = Normal()
        case 'moons': target = Moons()
        case _: target = PointCloud(args.target)

    X = target.sample(args.n)

    start = time.time()
    match args.algorithm:
        case 'unsmoothed': z_list = plain_unsmoothed(X, args.S, args.K)
        case 'smoothed': z_list = plain_smoothed(X, args.S, args.K, args.sigma, args.M)
        case 'smoothed_from_T': z_list = smoothed_from_T(X, args.S, args.K, args.sigma, args.M, args.T)
        case 'unsmoothedDEANN': z_list = unsmoothedDEANN(X, args.S, args.K, args.k, args.nprobe, args.l)
        case 'smoothedDEANN': z_list = smoothedDEANN(X, args.S, args.K, args.sigma, args.M, args.k, args.nprobe, args.l)
        case 'smoothedDEANN_from_T': z_list = smoothedDEANN_from_T(X, args.S, args.K, args.sigma, args.M, args.k, args.nprobe, args.l, args.T)
    end = time.time()
    runtime = end-start
    print('Runtime:', runtime, 'seconds')

    plot_titles_2d = {
        'unsmoothed': 'Generating training samples by flowing base samples through a velocity field using the unsmoothed score',
        'smoothed': 'Generating novel samples by flowing base samples through a velocity field using the smoothed score',
        'smoothed_from_T': 'Generating novel samples by flowing samples from a Mixture of Gaussians at time T through a velocity field using the smoothed score',
        'unsmoothedDEANN': 'Generating training samples by flowing samples through an through an approximation of the unsmoothed score using DEANN',
        'smoothedDEANN': 'Generating novel samples by flowing base samples through an approximation of the smoothed score using DEANN'
    }
    caption = 'args: '
    for arg in vars(args):
        caption += (f'%s: %s; ' % (arg, getattr(args, arg)))

    caption += f'\nRuntime: %ss' % (runtime)
    # caption += f'\nNumber of unique sampled points: {len(np.unique(z_list, axis=0))}'
    print(f'Number of unique sampled points: {len(np.unique(z_list, axis=0))}')

    match args.plot:
        case 'trajectories': trajectory_animated(X, z_list, plot_titles_2d[args.algorithm], caption, args.algorithm, args.target)
        case 'scatterplot': scatterplot(X, z_list[:, -1, :], plot_titles_2d[args.algorithm], caption, args.algorithm, args.target)
        case 'scatterplot-3d': plot_pointcloud(X, z_list[:, -1, :], args.target, args.algorithm, caption)
        case 'trajectories-3d': plot_pointcloud(X, np.reshape(z_list, (z_list.shape[1] * z_list.shape[0], z_list.shape[2])), args.target, args.algorithm, caption)

if __name__ == '__main__':

    class args:
        def __init__(self, comb):
            self.N=comb[0];
            self.n=comb[1];
            self.K=comb[2];
            self.S=comb[3];
            self.target=comb[4];
            self.plot=comb[5];
            self.algorithm=comb[6]; 
            self.sigma=comb[7];
            self.M=comb[8];
            # self.nprobe=comb[9];
            # self.k=comb[10];
            # self.l=comb[11];
            self.T=comb[9];

    grid = [
        [1000], # Number of held-out samples
        [2], # Number of training samples
        [2], # Number of samples to be generated
        [100], # Number of steps
        ['line'], # Target
        ['trajectories'], # Plotting strategy
        ['smoothed'], # Algorithm
        [1], # sigma
        [2], # M
        # [5], # nprobe
        # [15], # k
        # [15], # l
        [0.95] # T
    ]
    combinations = list(product(*grid))
    for idx in range(len(combinations)):
        print(f'Experiment no. {idx+1} of {len(combinations)}')
        main(args(combinations[idx]))