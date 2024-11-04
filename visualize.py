import os
import argparse
import numpy as np

import polyscope as ps
ps.init()
ps.set_ground_plane_mode('none')

parser  = argparse.ArgumentParser(
    prog = "Visualizer for Closed-Form Diffusion Models\n",
)

parser.add_argument(
    'algorithm',
    metavar='A',
    type=str, 
    choices=['smoothed', 'smoothed_from_T', 'smoothedDEANN', 'smoothedDEANN_from_T'],
    help='algorithm used',
)

parser.add_argument(
    'target', 
    metavar='T',
    help='target distribution',
    type=str
)

args = parser.parse_args()

target = args.target
algorithm = args.algorithm

path = f'results/3d/%s/%s' % (target, algorithm)

# Get the list of all directories
dirs = sorted(os.listdir(path))

n, K = None, None
for i, dir in enumerate(dirs):
    if not dir.startswith('.') and not dir.startswith('imgs'):
        print(f'Folder name: {dir}')
        with open(f'%s/%s/args.txt' % (path, dir), 'r') as txt:
            text = txt.read()
            n, K = int(text.split(';')[1][4:]), int(text.split(';')[2][4:])
            print(text, '\n' + ''.join(['-' for _ in range(10)]))
    
        cloud = np.loadtxt(f'%s/%s/array.txt' % (path, dir), dtype=np.longdouble)

        train_cloud = ps.register_point_cloud(
            name = 'train',
            points = cloud[:n],
            color=(255,0,0),
            radius=0.01
        )
        sampled_cloud = ps.register_point_cloud(
            name = 'sampled',
            points = cloud[n:],
            color=(255/255, 191/255, 0/255)
        )
        ps.show()
