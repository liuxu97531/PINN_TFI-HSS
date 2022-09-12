import numpy as np

import matplotlib.pyplot as plt


def uniform(x, start, end, rmin=0., range=0.5):
    if end - start <= 1:
        return
    mid = (start + end) // 2
    x[start:mid] = (x[start:mid] - rmin) / 2 + rmin
    x[mid:end] = (x[mid:end] - rmin) / 2 + rmin + range
    uniform(x, start, mid, rmin=rmin, range=range / 2)
    uniform(x, mid, end, rmin=range + rmin, range=range / 2)


def uniform2d(x, start, end, rminx=0., rminy=0., range=0.5):
    if end - start <= 1:
        return
    print(start, end, rminx, rminy, range)
    d, r = (end - start) // 4, (end - start) % 4
    r = np.array([(r + 3) // 4, (r + 2) // 4, (r + 1) // 4])
    np.random.shuffle(r)
    d1, d2, d3 = d + r[0], d + r[1], d + r[2]
    q1 = start + d1
    q2 = start + d1 + d2
    q3 = start + d1 + d2 + d3
    x[start:q1, 0] = (x[start:q1, 0] - rminx) / 2 + rminx
    x[start:q1, 1] = (x[start:q1, 1] - rminy) / 2 + rminy

    x[q1:q2, 0] = (x[q1:q2, 0] - rminx) / 2 + rminx + range
    x[q1:q2, 1] = (x[q1:q2, 1] - rminy) / 2 + rminy

    x[q2:q3, 0] = (x[q2:q3, 0] - rminx) / 2 + rminx
    x[q2:q3, 1] = (x[q2:q3, 1] - rminy) / 2 + rminy + range

    x[q3:end, 0] = (x[q3:end, 0] - rminx) / 2 + rminx + range
    x[q3:end, 1] = (x[q3:end, 1] - rminy) / 2 + rminy + range

    uniform2d(x, start, q1, rminx=rminx, rminy=rminy, range=range / 2)
    uniform2d(x, q1, q2, rminx=rminx + range, rminy=rminy, range=range / 2)
    uniform2d(x, q2, q3, rminx=rminx, rminy=rminy + range, range=range / 2)
    uniform2d(x, q3, end, rminx=range + rminx, rminy=rminy + range, range=range / 2)


def sample_uniform(N=40, x_end=10, y_end=10):
    x = np.linspace(0.0001, 0.9999, N).reshape(-1, 1)
    y = np.linspace(0.0001, 0.9999, N).reshape(-1, 1)
    points = np.concatenate([x, y], axis=1)
    uniform2d(points, start=0, end=N)

    X = x_end * points[:, 0]
    Y = y_end * points[:, 1]
    X = np.insert(X, 0, values=5, axis=0)
    Y = np.insert(Y, 0, values=0.25, axis=0)
    plt.scatter(X, Y)
    plt.show()
    # print('uniform sample:{}'.format(len(X)))
    # print('测点:')
    # print(X)
    # print(Y)
    return X, Y


x, y = sample_uniform(N = 80)
# print(x, y)