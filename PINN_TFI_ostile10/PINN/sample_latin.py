import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.pyplot as plt


def LHSample(D, bounds, N):
    '''
    :param D:参数个数
    :param bounds:参数对应范围（list）
    :param N:拉丁超立方层数
    :return:样本数据
    '''

    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size=1)[0]
        # print('temp', temp)
        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    # 对样本数据进行拉伸
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('范围出错')
        return None

    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result


def sample_latin(N=10, x_end=10, y_end=10):
    D = 2
    # N = 10
    bounds = [[0, x_end], [0, y_end]]
    xs = (bounds[0][1] - bounds[0][0]) / N
    ys = (bounds[1][1] - bounds[1][0]) / N
    ax = plt.gca()
    plt.ylim(bounds[1][0] - ys, bounds[1][1] + ys)
    plt.xlim(bounds[0][0] - xs, bounds[0][1] + xs)
    plt.grid()
    ax.xaxis.set_major_locator(MultipleLocator(xs))
    ax.yaxis.set_major_locator(MultipleLocator(ys))
    samples = LHSample(D, bounds, N)
    XY = np.array(samples)
    X = XY[:, 0]
    Y = XY[:, 1]
    X = np.insert(X, 0, values=5, axis=0)
    Y = np.insert(Y, 0, values=0.25, axis=0)
    # plt.scatter(X, Y)
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.show()
    # print('latin sample:{}'.format(len(X)))
    # print('测点:')
    # print(X)
    # print(Y)
    return X, Y
# from models import setup_seed
# setup_seed(0)
# sample_latin(N=10)
