import sys
sys.path.insert(0, " thesis") 

from os import listdir
from copy import deepcopy

from algorithms import smoothedDEANN_from_T, smoothed_from_T

import numpy as np
import point_cloud_utils as pcu
import polyscope as ps
ps.init()

dir = 'data/point_clouds/pc_gen_test'

def load_dataset():

    N = 5000
    pcs = None

    for file in sorted(listdir(dir)):
        print(file)
        v, f = pcu.load_mesh_vf(f'{dir}/{file}')
        count = deepcopy(N)
        while True:
            fid, bc = pcu.sample_mesh_poisson_disk(v, f, count)
            points = pcu.interpolate_barycentric_coords(f, fid, bc, v)
            print(f'Training points: {points.shape[0]}')
            if points.shape[0] >= (N+1):
                break
            count += 100*np.sign(N - points.shape[0])
        # print('Training set sampled')
        points = points[:N]
        pc = ((points - np.mean(points, axis=0)) / np.max(np.linalg.norm(points, axis=1)))
        pcs = pc[np.newaxis, :] if pcs is None else np.concatenate((pcs, pc[np.newaxis, :]), axis=0)
    
    np.save(f'{dir}/array.npy', pcs)
    return pcs


def main():

    '''
    Experimenting generating novel point clouds starting from the ModelNet dataset (here, only the 'table' class is used).
    '''

    if 'array.npy' not in listdir(dir):
        X = load_dataset()
    else:
        X = np.load(f'{dir}/array.npy')
    print(X.shape)

    z = smoothedDEANN_from_T(
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2]), 
        S = 1000, 
        K = 1,
        sigma = 0.08,
        M = 10,
        start_T = 0.95,
        k = 15,
        l = 15,
        nprobe = 5
    )[-1, -1, :]

    ps.register_point_cloud(
        name = 'cloud',
        points = z.reshape(5000, 3)
    )
    ps.show()


if __name__ == '__main__':
    main()