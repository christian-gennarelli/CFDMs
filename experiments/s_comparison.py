import sys
sys.path.insert(0, " thesis") 

import datetime
import argparse
from os import makedirs, listdir

import numpy as np
import point_cloud_utils as pcu
import matplotlib.pyplot as plt
import polyscope as ps
ps.init()

from algorithms import plain_smoothed
from targets import PointCloud
from w2_distance import w2_dist

def main(S_list, target_name):

    '''
    Experiment showing how an increasing number of steps can make the approximation
    of the velocity field followed by input points more accurate.
    '''

    n = 150

    target = PointCloud(target_name)
    X = target.sample(n)
    proxy = target.sample(500)

    w2_list = []
    for S in S_list:
        z = plain_smoothed(
            X = X,
            S = S,
            K = 15000,
            sigma = 0.08,
            M = 10
        )[:, -1, :]
        ps.register_point_cloud(
            name = f'cloud_S{S}',
            points = z,
        )

        dt = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        dir = f' results/experiments/S_comparison/%s/%s' % (target_name, dt)
        makedirs(dir)

        np.savetxt(f'%s/array.txt' % (dir), np.vstack([X, z]), fmt='%s')

        w2_distance = w2_dist(proxy, z)
        w2_list.append(w2_distance)

        with open(f'%s/info.txt' % (dir), 'w') as txt:
            txt.write(f'S: {S}; n: {X.shape[0]}\nW2 distance: {w2_distance}')
    
    plt.plot(S_list, w2_list)

def view(target_name):
    dir = f' results/experiments/S_comparison/{target_name}'

    for el in sorted(listdir(dir)):
        if not el.startswith('.'):
            cloud = np.loadtxt(f'{dir}/{el}/array.txt', dtype=np.longdouble)
            with open(f'{dir}/{el}/info.txt', 'r') as txt:
                args = txt.readline()
                args_split = args.split(';')
                S, n = int(args_split[0][3:]), int(args_split[2][3:])

                distance = txt.readline()
                print(args, distance)

            ps.register_point_cloud(
                name = f'train_cloud',
                points = cloud[:n],
                color = (1, 0, 0),
                radius = 0.01
            )
            ps.register_point_cloud(
                name = f'cloud_S{S}',
                points = cloud[n:],
                color = (1, 191/255, 0),
            )
            ps.show()
            ps.remove_point_cloud(f'cloud_S{S}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest = 'method'
    )

    view_subparser = subparsers.add_parser(
        'view',
        help = 'view results of the experiments'
    )
    view_subparser.add_argument(
        '-t',
        '--target',
        required = True,
        type = str,
        help = 'target distribution' 
    )

    main_subparser = subparsers.add_parser(
        'main',
        help = 'perform the experiment'
    )
    def check_valid_S(S_list):
        for S in S_list:
            if int(S) <= 0: return False
        return True
    main_subparser.add_argument(
        '-S',
        '--S_list',
        nargs = '+',
        type = check_valid_S,
        required = True,
        help = 'list of number of steps to perform'
    )
    main_subparser.add_argument(
        '-t',
        '--target',
        required = True,
        type = str,
        help = 'target distribution' 
    )

    args = parser.parse_args()

    match args.method:
        case 'view': view(args.target)
        case 'main': main(args.S_list, args.target)