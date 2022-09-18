import matplotlib.pyplot as plt
import numpy as np
import torch

def obs_sample(positions=torch.Tensor([[5., 5.], [8., 8.], [3, 3], [7, 2], [5, 8],
                                       [2, 3], [3, 6], [3, 8], [7, 4], [8, 6]]), n=9, n_ava=0, n_b=4,
               uniform_point=True, random=False):
    x = np.linspace(1.1, 8.9, n_ava)
    y = np.linspace(1.1, 8.9, n_ava)
    xs, ys = np.meshgrid(x, y)
    xs = xs.flatten().tolist()
    ys = ys.flatten().tolist()
    n_ava = len(xs)

    # 边界
    x_b = np.linspace(0.5, 9.5, n_b).tolist()
    y_b = np.linspace(0.5, 9.5, n_b).tolist()

    xs.extend(x_b)
    ys.extend(0.5 * np.ones_like(x_b))
    xs.extend(x_b)
    ys.extend(9.5 * np.ones_like(x_b))

    xs.extend(0.5 * np.ones_like(y_b))
    ys.extend(y_b)
    xs.extend(9.5 * np.ones_like(y_b))
    ys.extend(y_b)
    # xs.extend([2, 8, 2, 8])
    # ys.extend([0.5, 0.5, 9.5, 9.5])
    n_bc = len(xs) - n_ava

    # 小孔附近采点
    xs.extend([5])
    ys.extend([0.8])
    n_osti = len(xs) - n_bc - n_ava

    for i in range(len(positions)):
        xs.append(positions[i, 0].item())
        ys.append(positions[i, 1].item())
    n_total = len(xs)
    n_con = n_total - n_osti - n_bc - n_ava

    x_obs = xs
    y_obs = ys

    if uniform_point:
        nx = 10 / (n - 1)
        x = np.linspace(0, 10, n)
        y = np.linspace(0, 10, n)
        xs, ys = np.meshgrid(x, y)
        xs = xs.flatten().tolist()
        ys = ys.flatten().tolist()
        for i in range(1, n):
            plt.axhline(xs[i], c='k')
            plt.axvline(xs[i], c='k')
        plt.scatter(x_obs, y_obs, s=20, c='r', marker='x')
        plt.ylim(0, 10)
        plt.xlim(0., 10.)
        plt.show()

        n_x = [0] * len(x_obs)
        n_y = [0] * len(x_obs)
        n_loc = [0] * len(x_obs)

        for i in range(len(x_obs)):
            n_x[i] = int(x_obs[i] / nx) + 1
            n_y[i] = int(y_obs[i] / nx)
            n_loc[i] = (n_y[i]) * (n - 1) + n_x[i]
        n_loc = list(set(n_loc))  # 删除相同元素
        x = [i for i in range(1, (n - 1) ** 2 + 1)]
        for i in n_loc:
            x.remove(i)
        m_x = [0] * len(x)
        m_y = [0] * len(x)
        out_x = []
        out_y = []
        for i, j in enumerate(x):
            m_x[i] = j % (n - 1)
            m_y[i] = j // (n - 1)
            if random:
                if m_x[i] != 0:
                    out_x.append(m_x[i] * nx - np.random.rand(1).item() * (nx / 2))
                    out_y.append(m_y[i] * nx + np.random.rand(1).item() * (nx / 2))
                else:
                    out_x.append((n - 1) * nx - (nx / 2))
                    out_y.append((m_y[i] - 1) * nx + (nx / 2))
            else:
                # 取中间
                if m_x[i] != 0:
                    out_x.append(m_x[i] * nx - (nx / 2))
                    out_y.append(m_y[i] * nx + (nx / 2))
                else:
                    out_x.append((n - 1) * nx - (nx / 2))
                    out_y.append((m_y[i] - 1) * nx + (nx / 2))
        x_obs.extend(out_x)
        y_obs.extend(out_y)
        # for i in range(1, n):
        #     plt.axhline(xs[i], c='k')
        #     plt.axvline(xs[i], c='k')
        # plt.scatter(x_obs, y_obs, s=20, c='r', marker='x')
        # plt.ylim(0, 10)
        # plt.xlim(0., 10.)
        # plt.show()
    else:
        nx = 10 / (n - 1)
        x = np.linspace(0, 10, n)
        y = np.linspace(0, 10, n)
        xs, ys = np.meshgrid(x, y)
        xs = xs.flatten().tolist()
        ys = ys.flatten().tolist()
        for i in range(1, n):
            plt.axhline(xs[i], c='k')
            plt.axvline(xs[i], c='k')
        # plt.scatter(x_obs, y_obs, s=20, c='r', marker='x')
        # plt.ylim(0, 10)
        # plt.xlim(0., 10.)
        # plt.show()
    print('测点数量:{},均匀点:{},边界点:{},小孔:{},组件中心:{},额外点{}'.format(len(x_obs), n_ava, n_bc, n_osti, n_con,
                                                             len(x_obs) - n_total))
    seq = [i for i in range(len(x_obs))]

    seq_exp_con = seq[n_ava + n_bc + n_osti + n_con:]
    return x_obs, y_obs, seq_exp_con

# obs_sample(n = 8)