import argparse

'''
Key arguments:
1. Number of held-out samples
2. Number of training samples used to generate a sample
3. Number of samples to generate
4. Number of features
5. Numbers of steps to perform
6. Target distribution
7. Algorithm
8. Number of noise levels
'''

def parse_arguments():
    # Instantiate parser
    parser  = argparse.ArgumentParser(
        prog = "CFDM model\n",
        # description='Parsing key arguments for CFDM model'
    )

    # Positional arguments
    key_args = [
        ['N', int, 'number of held-out samples', 5000],
        ['n', int, 'number of training samples', 500],
        ['K', int, 'number of samples to generate', 10],
        ['S', int, 'number of steps to perform', 150],
        ['target', 'target distribution', ['checkerboard', 'spirals', 'std_normal', 'line', 'moons', 'hand', 'car_body', 'car_wheel_cap', 'gear']],
        ["plot", 'specify how to plot sampled points', ['scatterplot', 'trajectories', 'scatterplot-varying-sigma', 'scatterplot-3d', 'trajectories-3d']]
    ]
    # Parse positional arguments
    for arg in key_args:
        if len(arg) > 3: parser.add_argument(arg[0], type=arg[1], help=arg[2], default=arg[3])
        else: parser.add_argument(arg[0], help=arg[1], choices=arg[2])


    subparsers = parser.add_subparsers(
        dest='algorithm',
        help='algorithm to be performed'
    )

    # Possible algorithms to be performed
    algos = {
        'unsmoothed': {
            'help': 'flow a random initial point through the unsmoothed score',
            'args': [],
        },
        'smoothed':{
            'help': 'flow a random initial point through the smoothed score',
            'args': [
                ['sigma', float, 'smoothing parameter'],
                ['M', int, 'number of noise levels']
            ]
        },
        'smoothed_from_T': {
            'help': 'flow a random initial point through the unsmoothed score from T=0 to T=start_T, then through the smoothed score from T=start_T to T=1',
            'args': [
                ['sigma', float, 'smoothing parameter'],
                ['M', int, 'number of noise levels'],
                ['start_T', float, 'starting timestep']
            ]
        },
        'unsmoothedDEANN': {
            'help': 'flow a random initial point through the unsmoothed score from T=0 to T=1',
            'args': [
                ['nlist', int, 'number of Voronoi cells to divide the space in'],
                ['m', int, 'number of subvectors in which every vector will be split into (note: m must be chosen so that d % m = 0, otherwise we leave some dimensions unused)'],
                ['nbits', int, '2**nbits is the number of centroids in each subspace, thus each centroid for each subspace will require nbits bits to be represented'],
                ['nprobe', int, 'number of neighbouring Voronoi cells to look at when searching for the nearest neighbours'],
                ['k', int, 'number of nearest neighbours'],
                ['l', int, 'number of elements from the remainder of the dataset'],
            ]
        },
        'smoothedDEANN': {
            'help': 'flow a random initial sample through the smoothed score from T=0 to T=1',
            'args': [
                ['sigma', float, 'smoothing parameter'],
                ['M', int, 'number of noise levels'],
                ['k', int, 'number of nearest neighbours'],
                ['l', int, 'number of elements from the remainder of the dataset'],
                ['nprobe', int, 'number of neighbouring Voronoi cells to look at']
            ]
        },
        'smoothedDEANN_from_T':{
            'help': 'flow a random initial point through the approximation of the unsmoothed score from T=0 to T=start_T, then through the approximation of the smoothed score from T=start_T to T=1',
            'args': [
                ['sigma', float, 'smoothing parameter'],
                ['M', int, 'number of noise levels'],
                ['k', int, 'number of nearest neighbours'],
                ['l', int, 'number of elements from the remainder of the dataset'],
                ['nprobe', int, 'number of neighbouring Voronoi cells to look at'],
                ['T', float, 'starting timestep']
            ]
        }
    }

    for algo, desc in algos.items():
        # Instantiate new subparser
        subparser = subparsers.add_parser(
            algo,
            help=desc['help']
        )
        # Add arguments to the subparser
        for arg in desc['args']:
            subparser.add_argument(
                arg[0], type=arg[1], help=arg[2]
        )    

    args = parser.parse_args()

    if args.algorithm is None:
        print(f'Error: you must specify an algorithm...')
    
    return args

if __name__ == '__main__':
    args = parse_arguments()