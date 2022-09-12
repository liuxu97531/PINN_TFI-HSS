import scipy
import numpy as np

def ostile(T, nx, ny, n1, n2):

    n = (nx) * (ny)  # number of unknowns
    d = np.ones(n)  # diagonals
    u = np.ones(n - nx)
    d0 = d.copy() * -4
    b = np.zeros(n)  # RHS

    d1_upper = d.copy()
    d1_lower = d.copy()
    d2_lower = u.copy()
    d2_upper = u.copy()

    # 下边界
    d2_upper[:nx] = 2
    d2_upper[n1:n2] = 1
    # 左边界
    d1_upper[::nx] = 2
    d1_lower[nx - 1::nx] = 0

    # 右边界
    d1_upper[nx - 1::nx] = 0
    d1_lower[nx - 2::nx] = 2
    # 上边界
    d2_lower[n - 2 * nx:] = 2
    b[n1:n2] = -T  # 下边
    A = scipy.sparse.diags([d0, d1_upper, d1_lower, d2_upper, d2_lower], [0, 1, -1, nx, -nx], format='csc')
    return A,b
def around(T, nx, ny):
    # 四周散热
    n = (nx) * (ny)  # number of unknowns
    d = np.ones(n)  # diagonals
    u = np.ones(n - nx)
    d0 = d.copy() * -4
    b = np.zeros(n)  # RHS

    d1_upper = d.copy()
    d1_lower = d.copy()
    d2_lower = u.copy()
    d2_upper = u.copy()

    # 左边界
    d1_lower[nx - 1::nx] = 0

    # # 右边界
    d1_upper[nx - 1::nx] = 0

    b[:nx] = -T  # 下边
    b[nx::nx] = -T  # 左边
    b[2 * nx - 1::nx] = -T  # 下边
    b[-nx:] = -T  # 上
    A = scipy.sparse.diags([d0, d1_upper, d1_lower, d2_upper, d2_lower], [0, 1, -1, nx, -nx], format='csc')
    return A,b
def down(T, nx, ny):
    # 底边散热
    n = (nx) * (ny)  # number of unknowns
    d = np.ones(n)  # diagonals
    u = np.ones(n - nx)
    d0 = d.copy() * -4
    b = np.zeros(n)  # RHS

    d1_upper = d.copy()
    d1_lower = d.copy()
    d2_lower = u.copy()
    d2_upper = u.copy()

    # 左边界
    d1_upper[::nx] = 2
    d1_lower[nx - 1::nx] = 0

    # 右边界
    d1_upper[nx - 1::nx] = 0
    d1_lower[nx - 2::nx] = 2
    # 上边界
    d2_lower[n - 2 * nx:] = 2

    b[:nx] = -T  # 下边
    A = scipy.sparse.diags([d0, d1_upper, d1_lower, d2_upper, d2_lower], [0, 1, -1, nx, -nx], format='csc')
    return A,b

def fxy(x, y, positions, units, phi):  # 泊松方程右边的函数
    out = 0
    for i in range(len(phi)):
        out += phi[i] * (np.tanh(1e4 * (x - positions[i, 0] + units[i, 0] / 2)) + 1) \
               * (np.tanh(1e4 * (-x + positions[i, 0] + units[i, 0] / 2)) + 1) * \
               (np.tanh(1e4 * (y - positions[i, 1] + units[i, 1] / 2)) + 1) \
               * (np.tanh(1e4 * (-y + positions[i, 1] + units[i, 1] / 2)) + 1) / (16)
    f2 = out
    return f2


def fxy_inverse(x, y, positions, units, h):  # 泊松方程右边的函数
    out = []
    out.append(((np.tanh(1e4 * (x - positions[0] + units[0] / 2)) + 1) \
                * (np.tanh(1e4 * (-x + positions[0] + units[0] / 2)) + 1) * \
                (np.tanh(1e4 * (y - positions[1] + units[1] / 2)) + 1) \
                * (np.tanh(1e4 * (-y + positions[1] + units[1] / 2)) + 1) / (16)).flatten() * h ** 2)
    f2 = out
    return f2