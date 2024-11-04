import sys
sys.path.insert(0, " thesis") 

import argparse

import numpy as np
import polyscope as ps
ps.set_ground_plane_mode('none')
ps.init()

from targets import PointCloud

def main(args):
    X = PointCloud(args.target).sample(1000)
    start_T = args.start_T

    normal = np.random.multivariate_normal(
        mean=np.zeros(3),
        cov=np.eye(3),
        size=150
    )

    tX = start_T * X
    z_list = None
    for _ in range(1000):
        random_idx = np.random.randint(0, X.shape[0], 1)
        tx = (tX[random_idx, :]).squeeze()
        z = np.random.multivariate_normal(
            mean=tx,
            cov=np.eye(X.shape[1])*((1-start_T)**2)
        ).reshape(1, -1)
        z_list = z if z_list is None else np.concatenate((z_list, z))

    ps.register_point_cloud(
        name = 'base',
        points = normal
    )
    ps.register_point_cloud(
        name = 'train',
        points = X
    )
    ps.register_point_cloud(
        name = 'rescaled',
        points = z_list
    )
    ps.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--target',
        nargs = '+',
        type = str,
        choices = ['bed', 'airplane', 'bed', 'bench', 'bottle', 'car_body', 'car', 'chair', 'cup', 'steering_wheel', 'table'],
        required = True,
        help = 'target distribution'
    )

    def check_valid_T(T):
        if 0 <= float(T) <= 1: return True
        else: return False
    parser.add_argument(
        '-T',
        '--start_T',
        type = check_valid_T,
        required = True,
        help = 'timestep at which unsmoothed samples are drawn'
    )
    args = parser.parse_args()

    main(args)