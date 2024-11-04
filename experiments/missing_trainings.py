import sys
sys.path.insert(0, "/Users/genna/Documents/ACSAI/Closed-Form DM/thesis")

from algorithms import smoothed_from_T, plain_smoothed
from targets import PointCloud

import numpy as np
import polyscope as ps; ps.init()

def main():
    
    target = PointCloud('bed')
    X = target.sample(500)[np.r_[0:150, 250:500]]
    ps.register_point_cloud(
        name = 'train',
        points = X
    )

    z = plain_smoothed(
        X = X,
        S = 100,
        K = 10000,
        sigma = 0.08,
        M = 10,
    )[:, -1, :]
    ps.register_point_cloud(
        name = 'smoothed',
        points = z
    )

    z = smoothed_from_T(
        X = X,
        S = 100,
        K = 10000,
        sigma = 0.08,
        M = 10,
        start_T = 0.95
    )[:, -1, :]
    ps.register_point_cloud(
        name = 'smoothed_from_T',
        points = z
    )

    ps.show()

if __name__ == '__main__':
    main()