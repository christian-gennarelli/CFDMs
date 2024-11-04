import time

import numpy as np
import polyscope as ps
ps.init()
ps.set_ground_plane_mode('none')

def main():

    '''
    Create an animation of a sampled point cloud rotating on a particular axis.
    '''

    obj = '/Users/genna/Documents/ACSAI/Closed-Form DM/thesis/results/3d/cup/smoothed_from_T/18-09-2024-10-31-06/'

    cloud = np.loadtxt(f'{obj}/array.txt')
    with open(f'{obj}/args.txt', 'r') as txt:
        text = txt.read()
        n, K = int(text.split(';')[1][4:]), int(text.split(';')[2][4:])
        print(text, '\n' + ''.join(['-' for _ in range(10)]))
    
    train_points = cloud[:n]
    sampled_points = cloud[n:]
    ps.register_point_cloud(
        name = 'sampled',
        points = sampled_points,
        color = (255/255, 191/255, 0)
    )
    ps.register_point_cloud(
        name = 'train',
        points = train_points,
        color = (1, 0, 0),
        radius = 0.01
    )

    while(not ps.window_requests_close()):

        theta = np.radians(5)

        # Rotation around x-axis #
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        # Rotation around y-axis #
        rotation_matrix_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Rotation around z-axis #
        rotation_matrix_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        sampled_points = sampled_points @ rotation_matrix_z.T
        train_points = train_points @ rotation_matrix_z.T
        ps.register_point_cloud(
            name = 'sampled',
            points = sampled_points
        )
        ps.register_point_cloud(
            name = 'train',
            points = train_points
        )

        ps.frame_tick()
        time.sleep(0.05)

def animate_train():
    
    points = np.loadtxt('array.txt')
    
    while(not ps.window_requests_close()):

        theta = np.radians(5)

        # Rotation around x-axis #
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        # Rotation around y-axis #
        rotation_matrix_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Rotation around z-axis #
        rotation_matrix_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        points = points @ rotation_matrix_z.T
        ps.register_point_cloud(
            name = 'train',
            points = points
        )

        ps.frame_tick()
        time.sleep(0.05)

if __name__ == '__main__':
    main()