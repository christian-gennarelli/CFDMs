import numpy as np
import polyscope as ps

ps.init()
ps.set_ground_plane_mode('none')

dir = 'results/3d/bed/smoothed/09-09-2024-12-48-45'
n, K = None, None
with open(f'{dir}/args.txt', 'r') as txt:
    text = txt.read()
    n, K = int(text.split(';')[1][4:]), int(text.split(';')[2][4:])
    print(text, '\n' + ''.join(['-' for _ in range(10)]))
    
cloud = np.loadtxt(f'{dir}/array.txt', dtype=np.longdouble)

train_cloud = ps.register_point_cloud(
    name = 'train',
    points = cloud[:n],
    color = (1, 0, 0),
    radius = 0.01
)
sampled_cloud = ps.register_point_cloud(
    name = 'sampled',
    points = cloud[n:],
    color=(255/255, 191/255, 0/255),
)
ps.show()