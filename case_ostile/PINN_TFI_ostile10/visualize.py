import scipy.io as sio
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
from multiprocessing import Pool
import sys
import scipy.io as scio

sys.path.append('E:/pycharm_doc/reconstrution_PINN_benchmark/Inverse_ostiole/PINN')

from Parser_PINN import get_parser
import torch
import os
from net import Net
import numpy as np
from sample_latin import sample_latin
from sample_uniform import sample_uniform
from models import setup_seed


def plot_mat(mat_path, plot=True, save=False, worker=None, figkwargs={"figsize": (12, 5)}, ):
    mat_path = Path(mat_path)
    assert mat_path.exists(), "Input path does not exist!"
    if mat_path.is_dir():
        plot_dir(mat_path, save, worker)
        return
    mat = sio.loadmat(mat_path)
    xs, ys, u, F = mat["xs"], mat["ys"], mat["u"], mat["F"]

    fig = plt.figure(**figkwargs)
    plt.subplot(121)
    img = plt.pcolormesh(xs, ys, u, shading='auto', cmap='jet')
    plt.colorbar(img)
    plt.axis("image")
    plt.title("U")
    plt.subplot(122)
    img = plt.pcolormesh(xs, ys, F, shading='auto')
    plt.colorbar(img)
    plt.axis("image")
    plt.title("F")

    if plot:
        plt.show()
    if save:  # save png
        if save is True:
            img_path = mat_path.with_suffix(".png")
        else:  # save is path
            img_path = Path(save)
            if img_path.is_dir():  # save is dir
                img_path = (img_path / mat_path.name).with_suffix(".png")
        fig.savefig(img_path, dpi=100)
        plt.close()


def plot_dir(path, out, worker):
    path = Path(path)
    assert path.is_dir(), "Error! Arg path must be a dir."
    if out is None:
        out = True
    else:
        out = Path(out)
        print(out.absolute())
        if out.exists():
            assert Path(out).is_dir(), "Error! Arg out must be a dir."
        else:
            out.mkdir(parents=True)

    with Pool(worker) as pool:
        plot_mat_p = partial(plot_mat, plot=False, save=out)
        pool_iter = pool.imap_unordered(plot_mat_p, path.glob("*.mat"))
        for _ in tqdm.tqdm(
                pool_iter, desc=f"{pool._processes} workers's running"
        ):
            pass


def point_T(x_obs, y_obs, dataFile):
    data = scio.loadmat(dataFile)
    x_data = torch.from_numpy(data['xs'].reshape(-1, 1)).float().cuda()
    y_data = torch.from_numpy(data['ys'].reshape(-1, 1)).float().cuda()
    u_data = torch.from_numpy(data['u'].reshape(-1, 1)).float().cuda()
    x_obs = [float('{:.3f}'.format(i)) for i in x_obs]
    y_obs = [float('{:.3f}'.format(i)) for i in y_obs]
    idx = []
    for i in range(len(x_obs)):
        idx.append(int((int(x_obs[i] / 0.025)) * 400 + y_obs[i] / 0.025))
    x_data, y_data, u_data = x_data[idx], y_data[idx], u_data[idx]
    return x_data, y_data, u_data


def rel_error(args, dataFile, file, num, Epoch, sample):
    if sample == 'uniform':
        x_obs, y_obs = sample_uniform(N=num - 1)
    elif sample == 'latin':
        x_obs, y_obs = sample_latin(N=num - 1)
    fig, ax = plt.subplots(1, 3, figsize=(30, 9))
    data = sio.loadmat(dataFile)
    x_data = torch.from_numpy(data['xs'].reshape(-1, 1)).float().cuda()
    y_data = torch.from_numpy(data['ys'].reshape(-1, 1)).float().cuda()
    u_data = data['u']
    s = data['xs'].shape
    x_s = data['xs']
    y_s = data['ys']

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    PINN = Net(seq_net=args.seq_net, activation=args.activation).to(device)
    # PINN.load_state_dict(torch.load('./result/base/PINN0_1.pth'))
    PINN.load_state_dict(torch.load('./result_var/{}/{}/PINN_inverse_{}_{}.pth'.format(sample, file, Epoch, num)))
    z = PINN(torch.cat([x_data, y_data], dim=1))
    z_out = z.reshape(s)
    out = z_out.cpu().detach().numpy()
    img = ax[0].pcolormesh(x_s, y_s, out, cmap='jet')
    plt.colorbar(img, ax=ax[0])

    # ax[0].scatter(x_obs, y_obs, s=100, c='k', marker='x')

    error_rel = abs((out - u_data))
    img2 = ax[1].pcolormesh(x_s, y_s, error_rel, cmap='YlGnBu')
    cbar2 = plt.colorbar(img2, ax=ax[1])

    cbar2.mappable.set_clim(0, 10)
    ax[0].set_title('Pred T', fontsize=20)

    ax[1].set_title('Abs error({})'.format(file), fontsize=20)

    error_rel = abs((out - u_data) / out)
    img3 = ax[2].pcolormesh(x_s, y_s, error_rel, cmap='YlGnBu')
    cbar3 = plt.colorbar(img3, ax=ax[2])
    ax[2].set_title('Relative error', fontsize=20)
    plt.show()

    x = np.linspace(0.5, 9.5, 200)
    y = np.linspace(0.5, 9.5, 200)
    xs, ys = np.meshgrid(x, y)
    xs, ys = xs.flatten().tolist(), ys.flatten().tolist()
    x_error, y_error, u_data = point_T(xs, ys, dataFile=dataFile)

    u_pred = PINN(torch.cat([x_error, y_error], dim=1))
    error_rel = args.criterion(u_pred, u_data)
    error = error_rel
    print('损失：{}'.format(error))
    return error

def plot_loss(Epoch=20000, file='up', num=1):
    loss_history = np.load('./result/{}/loss{}_{}.npy'.format(file, Epoch, num))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.legend(('PDE loss', 'BC loss', 'Data loss', 'loss'), loc='best')
    plt.show()

    q_history = np.load('./result/{}/q{}_{}.npy'.format(file, Epoch, num))
    plt.plot(q_history)
    plt.savefig('./result/{}/q{}_{}.png'.format(file, Epoch, num))
    # np.save('./result/{}/q{}_{}.npy'.format(file, Epoch, num), q_history)
    plt.title('pred power')

    # plt.axhline(18, c='r', ls='--')
    plt.axhline(8, c='r', ls='--')

    # print(q_history[-1, 0].item(), ',', str((20 - q_history[-1, 0].item()) / 20 * 100) + '%')
    # print(q_history[-1, 1].item(), ',', str((30 - q_history[-1, 1].item()) / 30 * 100) + '%')
    print(q_history[-1, 0].item(), ',', str((5 - q_history[-1, 0].item()) / 5 * 100) + '%')
    print(q_history[-1, 1].item(), ',', str((10 - q_history[-1, 1].item()) / 10 * 100) + '%')
    print(q_history.shape)
    plt.show()


def plot_layout():
    positions = torch.Tensor([[5., 5.], [8., 8.], [3, 3], [7, 2], [5, 8],
                              [2, 3], [3, 6], [3, 8], [7, 4], [8, 6]])
    units = torch.Tensor([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                          [1, 1], [2, 2], [1, 1], [2, 2]])
    num = 10
    phi = [10] * num

    x = torch.linspace(0, 10, 400)
    y = torch.linspace(0, 10, 400)
    xs, ys = torch.meshgrid(x, y)
    s = xs.shape
    x_f = xs.reshape(-1, 1)
    y_f = ys.reshape(-1, 1)

    out = 0.
    for i in range(len(phi)):
        out += (torch.tanh(1e5 * (x_f - positions[i, 0] + units[i, 0] / 2)) + 1) \
               * (torch.tanh(1e5 * (-x_f + positions[i, 0] + units[i, 0] / 2)) + 1) * \
               (torch.tanh(1e5 * (y_f - positions[i, 1] + units[i, 1] / 2)) + 1) \
               * (torch.tanh(1e5 * (-y_f + positions[i, 1] + units[i, 1] / 2)) + 1) / (16)

    im = out.reshape(s)
    plt.pcolormesh(xs.cpu(), ys.cpu(), im.cpu())
    plt.show()


if __name__ == '__main__':
    # dataFile = './result/up0.mat'
    dataFile = './result/draft0.mat'
    plot_mat(mat_path=dataFile)

    setup_seed(0)

    sample = 'latin'
    # sample = 'uniform'
    # sample = 'mesh'
    # file = 'base'
    # file = 'up'
    file = 'down'
    # file = 'up_down'
    error = []
    dataFile = 'result/{}0.mat'.format(file)
    # plot_mat(mat_path=dataFile)
    # for i in [30]:
    for i in range(25, 30):
        print('{}个测点：'.format(i + 2))
        num = i + 2
        parser = get_parser()
        args = parser.parse_args()
        error_i = rel_error(args, dataFile=dataFile, file=file, num=num, Epoch=5000, sample=sample)
        error.append(error_i.item())
    print(error)
    # plot_loss(Epoch=5000, file=file, num=num)
