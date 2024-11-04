import datetime
from os import makedirs

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import polyscope as ps
import numpy as np
from targets import PointCloud

def scatterplot(X, z_list, title, caption, algorithm, target):

    fig, ax = plt.subplots()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    
    x, y = z_list.T
    ax.scatter(x, y, c='blue', label='Sampled points', alpha=.5)

    # Plot target distribution samples
    ax.scatter(X[:,0], X[:,1], c='red', label='Training points')

    fig.text(.5, .05, caption, fontsize='small', ha='center', wrap=True)
    fig.suptitle(title)
    ax.legend()

    # dt = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    # dir = f' results/2d/%s/%s' % (target, algorithm)
    # makedirs(dir, exist_ok=True)
    # plt.savefig(f'%s/%s.png' % (dir, dt), pad_inches = 0.5)

    plt.show()

def trajectory_animated(X, z_list, title, caption, algorithm, target):
    
    fig, ax = plt.subplots()

    # Plot target distribution samples
    ax.scatter(X[:,0], X[:,1], c='red', label='Training points')

    # colors = ["#%06x" % randint(0, 0xFFFFFF) for _ in range(len(z_list))]
    lines = [ax.plot([], [], marker='o', lw=2)[0] for _ in range(len(z_list))]

    # Initialization function to plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # Animation function to be called sequentially
    def animate(i):
        for line, z in zip(lines, z_list):
            x, y = z[:,0], z[:,1]
            line.set_data(x[:i+1], y[:i+1])
        if i == len(z) - 1: # Last frame
            ax.scatter(z_list[:,-1,0], z_list[:,-1,1], s=50, c='blue', zorder=10, label='Sampled points', alpha=0.2)
            ax.legend()
        return lines

    # Create the animation object
    interval = 50
    frames = len(z_list[1])
    ani = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=interval, repeat=False)

    fig.suptitle(title)
    fig.text(.5, .05, caption, fontsize='small', ha='center', wrap=True)
    plt.show()

    # Save animation as GIF
    writer = PillowWriter(fps=15,
                metadata=dict(artist='Christian Gennarelli'),
                bitrate=1800)
    
    dt = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    dir = f' results/2d/%s/%s' % (target, algorithm)
    makedirs(dir)
    ani.save(f'%s/%s.gif' % (dir, dt), writer=writer)

def plot_pointcloud(X, z_list, target, algorithm, caption):

    dir = f'results/3d/%s/%s/'

    ps.init()
    ps.set_ground_plane_mode('none')

    ps.register_point_cloud(
        name = 'train',
        points = X
    )
    ps.register_point_cloud(
        name = 'sampled',
        points = z_list
    )

    ps.show()

    dt = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    dir = f'results/3d/%s/%s/%s' % (target, algorithm, dt)
    makedirs(dir)

    np.savetxt(f'%s/array.txt' % (dir), np.vstack([X, z_list]), fmt='%s')
    
    with open(f'%s/args.txt' % (dir), 'w') as txt:
        txt.write(caption)

if __name__ == '__main__':
    plot_pointcloud(
        X = PointCloud('hand').sample(250),
        z_list = np.zeros((2,3)),
        target='hand',
        algorithm='smoothed',
        caption=''
    )