from os import listdir
from copy import deepcopy

from abc import ABC, abstractmethod
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
import point_cloud_utils as pcu
import polyscope as ps

rng = np.random.RandomState()

class Target(ABC):

    @abstractmethod
    def sample(self, num_samples):
        pass

### 2D distributions ###

class Checkerboard(Target):
    def sample(self, num_samples):
        a = np.random.rand(num_samples) * 8 - 4
        b = np.random.rand(num_samples) - np.random.randint(0, 2, num_samples) * 2
        c = b + (np.floor(a) % 2)
        return np.concatenate([a[:, None], c[:, None]], 1) * 2
    
class Line(Target):
    def sample(self, num_samples):
        x = [-1,1]
        y = [-1,-1]
        return np.stack((x, y), 1)

class Spirals(Target):
    def sample(self, num_samples):
        n = np.sqrt(np.random.rand(num_samples // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(num_samples // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(num_samples // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return x

class Normal(Target):
    def sample(self, num_samples):
        return np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=num_samples)

class Moons(Target):

    def sample(self, num_samples):
        x, _  = make_moons(num_samples, noise=.05)
        return np.array(x)

class SwissRoll(Target):
    def sample(self, num_samples):
        x, _ = make_swiss_roll(num_samples, noise=.05)
        return np.array(x)[:, [0,2]]

### 3D point clouds ###

class PointCloud(Target):

    def __init__(self, target_name):

        files = listdir('data/point_clouds')
        self.v, self.f = pcu.load_mesh_vf(f'data/point_clouds/{[file for file in files if file.startswith(target_name)][0]}')

    def sample(self, num_samples, accurate=True):

        fid, bc = pcu.sample_mesh_poisson_disk(self.v, self.f, num_samples)
        points = pcu.interpolate_barycentric_coords(self.f, fid, bc, self.v)
        if accurate:
            count = deepcopy(num_samples)
            while True:
                # print(f'Training points: {points.shape[0]}')
                if num_samples == points.shape[0]:
                    break
                count += 2*np.sign(num_samples - points.shape[0])
                fid, bc = pcu.sample_mesh_poisson_disk(self.v, self.f, count)
                points = pcu.interpolate_barycentric_coords(self.f, fid, bc, self.v)
        # print('Training set sampled')
        return ((points - np.mean(points, axis=0)) / np.max(np.linalg.norm(points, axis=1)))

if __name__ == '__main__':

    ps.init()
    points = PointCloud('cup').sample(150)
    ps.register_point_cloud(
        name = 'cloud',
        points = np.asarray(points)
    )
    
    ps.show()