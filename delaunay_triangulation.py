import numpy as np
import matplotlib.pyplot as plt

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
    for i in x:
        for j in y:
            if surface1(i, j) < surface2(i, j):
                point_cloud.append([i, j, surface1(i, j)])
                point_cloud.append([i, j, surface2(i, j)])
    return np.array(point_cloud)

def plot_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Point Cloud of the Closed Surface")
    plt.show()
    plt.savefig('point_cloud.png')

point_cloud = generate_point_cloud(x_range=(-1, 1), y_range=(-1, 1), points=200)
plot_point_cloud(point_cloud)