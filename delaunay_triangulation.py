import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
from scipy.spatial import Delaunay
from scipy.interpolate import griddata

def surface1(x, y):
    return 2*x**2 + 2*y**2

def surface2(x, y):
    return 2 * np.exp(-x**2 - y**2)

def generate_point_cloud(x_range, y_range, points=100):
    # Create a grid of x, y values
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.linspace(y_range[0], y_range[1], points)
    X, Y = np.meshgrid(x, y)
    point_cloud = []
    top = []
    bot = []
    for i in x:
        for j in y:
            if surface1(i, j) < surface2(i, j):
                point_cloud.append([i, j, surface1(i, j)])
                bot.append([i, j, surface1(i, j)])
                point_cloud.append([i, j, surface2(i, j)])
                top.append([i, j, surface2(i, j)])
    return np.array(top),np.array(bot),np.array(point_cloud)

def plot_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Point Cloud of the Closed Surface")
    plt.show()
    plt.savefig('point_cloud.png')

top_pc, bot_pc, point_cloud = generate_point_cloud(x_range=(-1, 1), y_range=(-1, 1), points=5)
plot_point_cloud(point_cloud)

def plot_triangulation(points, tri, file):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_zlim(0,2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()
    plt.savefig(file)

tri_top = Delaunay(top_pc[:, :2])
tri_bot = Delaunay(bot_pc[:, :2])

plot_triangulation(top_pc, tri_top.simplices, "top_tri.png")
plot_triangulation(bot_pc, tri_bot.simplices, "bot_tri.png")
'''def graham_scan(data):
    def polar_angle(p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    y_min_ind = np.argmin(data[:, 1])
    y_min = data[y_min_ind]
    polar_dict = {}
    for point in data:
        polar_dict.update({tuple(point[0:2]): polar_angle(y_min, point[0:2])})
    polar_dict = dict(sorted(polar_dict.items(), key=lambda item: item[1]))
    polar_points = [key for key in polar_dict]
    hull = []
    for point in polar_points:
        while len(hull) > 1 and np.cross(
            np.array(hull[-1]) - np.array(hull[-2]),
            np.array(point[0:2]) - np.array(hull[-1])
        ) <= 0:
            hull.pop()
        hull.append(point)
    hull.append(hull[0])
    return np.array(hull)

top_boundary = graham_scan(top_pc)
bot_boundary = graham_scan(bot_pc)'''

top_boundary = []
for i in range(len(tri_top.neighbors)):
    if -1 in tri_top.neighbors[i]: 
        top_boundary.append(tri_top.simplices[i])

top_boundary = np.array(top_boundary).flatten()
top_boundary = np.unique(top_boundary)
top_boundary = np.sort(top_boundary)

bot_boundary = []
for i in range(len(tri_bot.neighbors)):
    if -1 in tri_bot.neighbors[i]:
        bot_boundary.append(tri_bot.simplices[i])

bot_boundary = np.array(bot_boundary).flatten()
bot_boundary = np.unique(bot_boundary)
bot_boundary = np.sort(bot_boundary)

top_size = len(top_pc)
side_faces = []
for i in range(len(top_boundary) - 1):
    side_faces.append([top_boundary[i], top_boundary[i+1], bot_boundary[i]+top_size])
    side_faces.append([top_boundary[i+1], bot_boundary[i]+top_size, bot_boundary[i+1]+top_size])

shifted_tri_bot = tri_bot.simplices + top_size
print(shifted_tri_bot)
side_faces = np.array(side_faces).reshape(-1,3)
all_faces = np.vstack((tri_top.simplices, shifted_tri_bot, side_faces))
plot_triangulation(np.vstack((top_pc, bot_pc)), all_faces, "tri.png")

x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
grid_points = np.vstack([x_grid.flatten(), y_grid.flatten()]).T

z_values = griddata(point_cloud[:, :2], point_cloud[:, 2], grid_points, method='linear')

z_grid = z_values.reshape(x_grid.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(0, 2)
plt.title("Interpolated Volume Mesh Inside the Closed Surface")

plt.show()
plt.savefig('filled_surface.png')

points_2d = np.vstack([x_grid.flatten(), y_grid.flatten()]).T
delaunay = Delaunay(points_2d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(points_2d[:, 0], points_2d[:, 1], z_grid.flatten(), triangles=delaunay.simplices, cmap='viridis', edgecolor='none')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(0, 2)
plt.title("Surface Interpolated from Volume Mesh")

plt.show()
plt.savefig('rebuilt_surface.png')