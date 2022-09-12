import geo_utils.geo2d as gd2
import sympy
from sympy import Eq
import sys
sys.path.append('/mnt/share1/liuxu/reconstruction/PINN_reconstruction/PINN')
from net import Net
from models import PDE, is_neumann_boundary_x, is_neumann_boundary_y
import torch
import os
import matplotlib.pyplot as plt
from Parser_PINN import get_parser
import random
import numpy as np
import scipy.io as scio

import matplotlib.pyplot as plt
from sympy import Eq


def pde_loss(x, y, PINN, n_f, positions, units, phi, device):
    x_f = ((x[0] + x[1]) / 2 + (x[1] - x[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)

    y_f = ((y[0] + y[1]) / 2 + (y[1] - y[0]) *
           (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
           ).requires_grad_(True)

    u_f = PINN(torch.cat([x_f, y_f], dim=1))
    PDE_, out = PDE(u_f, x_f, y_f, positions, units, phi)
    mse_PDE = args.criterion(PDE_, torch.zeros_like(PDE_))
    return mse_PDE, out


def bc_dirichlet(x, y, PINN):
    u_b = (PINN(torch.cat([x, y], dim=1)))
    mse_BC = args.criterion(u_b, 80*torch.ones_like(u_b))
    return mse_BC

def bc_Neumann_x(x, y, PINN):
    u_b = (PINN(torch.cat([x, y], dim=1)))
    u_BC = is_neumann_boundary_x(u_b, x, y)
    mse_BC = args.criterion(u_BC, torch.zeros_like(u_BC))
    return mse_BC

def bc_Neumann_y(x, y, PINN):
    u_b = (PINN(torch.cat([x, y], dim=1)))
    u_BC = is_neumann_boundary_y(u_b, x, y)
    mse_BC = args.criterion(u_BC, torch.zeros_like(u_BC))
    return mse_BC

def train(args, Epoch):
    XS, YS = sympy.symbols('x y')
    rectan = gd2.Rectangle((0, 0), (10, 10))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    n_f = 8000
    n_b_bc = 2000
    n_f_loc = 1000

    dataFile = './result/base0.mat'
    data = scio.loadmat(dataFile)
    x_data = torch.from_numpy(data['xs'].reshape(-1, 1)).float().cuda()
    y_data = torch.from_numpy(data['ys'].reshape(-1, 1)).float().cuda()
    u_data = torch.from_numpy(data['u'].reshape(-1, 1)).float().cuda()

    idx = np.random.choice(160000, 10000, replace=False)
    x_data = x_data[idx]
    y_data = y_data[idx]
    u_data = u_data[idx]

    positions = torch.Tensor([[5., 5.], [8., 8.], [3, 3], [7, 2], [5, 8],
                              [2, 3], [3, 6], [3, 8], [7, 4], [8, 6]])
    units = torch.Tensor([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [1, 1], [2, 2]])
    phi = [10, 15, 10, 15, 10, 10, 15, 15, 15, 10]

    x_loc, y_loc = [], []
    for i in range(len(positions)):
        positions[i, 0] - units[i, 0] / 2
        x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
        y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

    PINN = Net(seq_net=[2, 50, 50, 50, 50, 50, 1], activation=args.activation).to(device)
    optimizer = args.optimizer(PINN.parameters(), args.lr)

    loss_history = []
    for epoch in range(Epoch):
        optimizer.zero_grad()
        # inside
        mse_PDE_c, _ = pde_loss(x=[0, 10], y=[0, 10], PINN=PINN, n_f=n_f, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE0, _ = pde_loss(x=x_loc[0], y=y_loc[0], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE1, _ = pde_loss(x=x_loc[1], y=y_loc[1], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE2, _ = pde_loss(x=x_loc[2], y=y_loc[2], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE3, _ = pde_loss(x=x_loc[3], y=y_loc[3], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE4, _ = pde_loss(x=x_loc[4], y=y_loc[4], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE5, _ = pde_loss(x=x_loc[5], y=y_loc[5], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE6, _ = pde_loss(x=x_loc[6], y=y_loc[6], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE7, _ = pde_loss(x=x_loc[7], y=y_loc[7], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE8, _ = pde_loss(x=x_loc[8], y=y_loc[8], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE9, _ = pde_loss(x=x_loc[9], y=y_loc[9], PINN=PINN, n_f=n_f_loc, positions=positions, units=units,
                               phi=phi, device=device)
        mse_PDE = mse_PDE_c + mse_PDE0 + mse_PDE1 + mse_PDE2 + mse_PDE3 + mse_PDE4 + \
                  mse_PDE5 + mse_PDE6 + mse_PDE7 + mse_PDE8 + mse_PDE9
        # boundary

        point_bc_ostile = rectan.sample_boundary(n_b_bc, criteria=Eq(YS, 0) & ((XS > 4.5) & (XS < 5.5)))
        point_bc_down = rectan.sample_boundary(n_b_bc, criteria=Eq(YS, 0) & ((XS < 4.5) | (XS > 5.5)))
        point_bc_left_right = rectan.sample_boundary(n_b_bc, criteria=Eq(XS, 0) | Eq(XS, 10))
        point_bc_up = rectan.sample_boundary(n_b_bc, criteria=Eq(YS, 10))

        x_bc_ostile = torch.from_numpy(point_bc_ostile['x']).float().cuda().requires_grad_(True)
        y_bc_ostile = torch.from_numpy(point_bc_ostile['y']).float().cuda().requires_grad_(True)
        x_bc_down = torch.from_numpy(point_bc_down['x']).float().cuda().requires_grad_(True)
        y_bc_down = torch.from_numpy(point_bc_down['y']).float().cuda().requires_grad_(True)
        x_bc_left_right = torch.from_numpy(point_bc_left_right['x']).float().cuda().requires_grad_(True)
        y_bc_left_right = torch.from_numpy(point_bc_left_right['y']).float().cuda().requires_grad_(True)
        x_bc_up = torch.from_numpy(point_bc_up['x']).float().cuda().requires_grad_(True)
        y_bc_up = torch.from_numpy(point_bc_up['y']).float().cuda().requires_grad_(True)

        mse_BC_1 = bc_dirichlet(x=x_bc_ostile, y=y_bc_ostile, PINN=PINN)
        mse_BC_2 = bc_Neumann_y(x=x_bc_down, y=y_bc_down, PINN=PINN)
        mse_BC_3 = bc_Neumann_y(x=x_bc_up, y=y_bc_up, PINN=PINN)
        mse_BC_4 = bc_Neumann_x(x=x_bc_left_right, y=y_bc_left_right, PINN=PINN)

        mse_BC = mse_BC_1 + mse_BC_2 + mse_BC_3 + mse_BC_4

        u_pred_data = PINN(torch.cat([x_data, y_data], dim=1))
        mse_data = args.criterion(u_pred_data, u_data)

        # loss

        loss = mse_PDE + mse_BC + 5*mse_data
        loss_history.append([mse_PDE.item(), mse_BC.item(), loss.item()])

        if epoch % 1000 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), loss.item()
                )
            )
        loss.backward()
        optimizer.step()



    # xx = torch.linspace(0, 10, 400).cpu()
    # yy = torch.linspace(0, 10, 400).cpu()
    # xs, ys = torch.meshgrid([xx, yy])
    # s1 = xs.shape
    # x1 = xs.reshape((-1, 1))
    # y1 = ys.reshape((-1, 1))
    # x_ = torch.cat([x1, y1], dim=1).to(device)
    # z = PINN(x_)
    # z_out = z.reshape(s1)
    # out = z_out.cpu().T.detach().numpy()[::-1, :]
    # plt.pcolormesh(xs, ys, out, cmap='jet')
    # plt.colorbar()
    # # plt.savefig('./result/{}/u{}_{}.png'.format(file, Epoch, num))
    # # np.save('./result/{}/u{}_{}.npy'.format(file, Epoch, num), out)
    # plt.show()

    # plt.cla()
    # plt.plot(loss_history)
    # plt.yscale('log')
    # plt.legend(('PDE loss', 'BC loss', 'loss'), loc='best')
    # plt.savefig('./result/{}/loss{}_{}.png'.format(file, Epoch, num))
    # plt.show()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args=args, Epoch=5000)
