import numpy as np
import point_cloud_utils as pcu

def w2_dist(a1, a2, eps=1e-6):
    M = pcu.pairwise_distances(a1, a2)
    w_a1 = np.ones(a1.shape[0])
    w_a2 = np.ones(a2.shape[0])
    P = pcu.sinkhorn(w_a1, w_a2, M, eps)
    return (M*P).sum()

if __name__ == '__main__':

    x = np.zeros(3)[:, np.newaxis]
    y = np.ones(3)[:, np.newaxis]

    print(f'x = {x}\ny = {y}')
    print(f'Wasserstein-2 distance between x and y: {w2_dist(x, y)}')

