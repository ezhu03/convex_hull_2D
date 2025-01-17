import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('mesh.dat', skiprows=1)

def graham_scan(data):
    def polar_angle(p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    y_min_ind = np.argmin(data[:, 1])
    y_min = data[y_min_ind]
    polar_dict = {}
    for point in data:
        polar_dict.update({tuple(point): polar_angle(y_min, point)})
    polar_dict = dict(sorted(polar_dict.items(), key=lambda item: item[1]))
    polar_points = [key for key in polar_dict]
    hull = []
    for point in polar_points:
        while len(hull) > 1 and np.cross(
            np.array(hull[-1]) - np.array(hull[-2]),
            np.array(point) - np.array(hull[-1])
        ) <= 0:
            hull.pop()
        hull.append(point)
    
    return np.array(hull)

def jarvis_march(data):
    x_min_ind = np.argmin(data[:, 0])
    x_min = data[x_min_ind]
    hull = []
    p=x_min
    n = len(data)
    while(True):
        hull.append(p)
        q = (p + 1) % n
        for r in data:
            if np.cross(np.array(r-p), np.array(q-p)) < 0:
                q = r
        p = q
        if np.allclose(p, x_min):
            break
    return np.array(hull)

def quick_hull(data):
    def find_side(p1, p2, point):
        return np.cross(p2 - p1, point - p1)
    x_min_ind = np.argmin(data[:, 0])
    x_min = data[x_min_ind]
    x_max_ind = np.argmax(data[:, 0])
    x_max = data[x_max_ind]
    top = []
    bot = []
    hull = []
    for point in data:
        if find_side(x_min,x_max,point) > 0:
            top.append(point)
        elif find_side(x_min,x_max,point) < 0:
            bot.append(point)
    def add_hull(data, p1, p2, side):
        if len(data) == 0:
            return
        def area(pt1, pt2, pt):
            return np.abs(np.cross(pt2 - pt1, pt - pt1))
        maximum = 0
        for point in data:
            if area(p1, p2, point) > maximum:
                maximum = area(p1, p2, point)
                max_point = point
        if maximum > 0:
            left = []
            right = []
            for point in data:
                if side > 0:
                    if find_side(p1, max_point, point) > 0:
                        left.append(point)
                    elif find_side(max_point, p2, point) > 0:
                        right.append(point)
                elif side < 0:
                    if find_side(p1, max_point, point) < 0:
                        left.append(point)
                    elif find_side(max_point, p2, point) < 0:
                        right.append(point)
            
            if side > 0:
                add_hull(left, p1, max_point, side)
                hull.append(max_point)
                add_hull(right, max_point, p2, side)
            elif side < 0:
                add_hull(right, max_point, p2, side)
                hull.append(max_point)
                add_hull(left, p1, max_point, side)
    hull.append(x_min)
    add_hull(top, x_min, x_max, 1)
    hull.append(x_max)
    add_hull(bot, x_min, x_max, -1)
    return np.array(hull)
def monotone_chain(data):
    def angle(p1, p2, p3):
        return np.cross(p2 - p1, p3 - p2)
    data = sorted(data, key=lambda item: item[0])
    top = []
    bot = []
    for point in data:
        while len(top) > 1 and angle(top[-2], top[-1], point) >= 0:
            top.pop()
        top.append(point)
    for point in reversed(data):
        while len(bot) > 1 and angle(bot[-2], bot[-1], point) >= 0:
            bot.pop()
        bot.append(point)
    return np.array(top[:-1] + bot[:-1])

hull = graham_scan(data)
plt.figure()
plt.plot(hull[:, 0], hull[:, 1],color='red')
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.show()
plt.savefig("graham_scan.png")
hull = jarvis_march(data)
plt.figure()
plt.plot(hull[:, 0], hull[:, 1],color='red')
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.show()
plt.savefig("jarvis_march.png")
hull = quick_hull(data)
plt.figure()
plt.plot(hull[:, 0], hull[:, 1],color='red')
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.show()
plt.savefig("quick_hull.png")
hull = monotone_chain(data)
plt.figure()
plt.plot(hull[:, 0], hull[:, 1], color='red')
plt.scatter(data[:, 0], data[:, 1], color='blue')
plt.show()
plt.savefig("monotone_chain.png")