import geo_utils.geo2d as gd2
import sympy
from sympy import Eq
import sys

sys.path.append('/mnt/share1/liuxu/reconstruction/PINN_reconstruction/PINN')

from net import Net
from models import PDE, is_neumann_boundary_x, is_neumann_boundary_y, PDE_inverse, setup_seed
import torch
import os
from Parser_PINN import get_parser
import numpy as np
import scipy.io as scio
from sample_latin import sample_latin
from sample_uniform import sample_uniform
from sample import obs_sample


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
    mse_BC = args.criterion(u_b, 80 * torch.ones_like(u_b))
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


def train(args, Epoch, num, x_obs, y_obs, data_weight, sample):
    if not os.path.exists('./result/{}'.format(sample)):
        os.makedirs('./result/{}'.format(sample))

    XS, YS = sympy.symbols('x y')
    rectan = gd2.Rectangle((0, 0), (10, 10))
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.cuda('cpu')
    dataFile = './result/up_down0.mat'
    data = scio.loadmat(dataFile)
    x_data = torch.from_numpy(data['xs'].reshape(-1, 1)).float().cuda()
    y_data = torch.from_numpy(data['ys'].reshape(-1, 1)).float().cuda()
    u_data = torch.from_numpy(data['u'].reshape(-1, 1)).float().cuda()
    n_f = 10000
    n_f_loc = 1000
    n_b_bc = 1000
    x_true, y_true, u_true = x_data, y_data, u_data
    idx = []
    for i in range(len(x_obs)):
        idx.append(int(int(x_obs[i] / 0.025) * 400 + (y_obs[i] / 0.025)))

    x_data, y_data, u_data = x_data[idx], y_data[idx], u_data[idx]

    positions = torch.Tensor([[5., 5.], [8., 8.], [3, 3], [7, 2], [5, 8],
                              [2, 3], [3, 6], [3, 8], [7, 4], [8, 6]])
    units = torch.Tensor([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [1, 1], [2, 2]])

    x_loc, y_loc = [], []
    for i in range(len(positions)):
        positions[i, 0] - units[i, 0] / 2
        x_loc.append([positions[i, 0] - units[i, 0] / 2, positions[i, 0] + units[i, 0] / 2])
        y_loc.append([positions[i, 1] - units[i, 1] / 2, positions[i, 1] + units[i, 1] / 2])

    PINN = Net(seq_net=[2, 50, 50, 50, 50, 1], activation=args.activation).to(device)
    PINN.load_state_dict(torch.load('./result/PINN0_1.pth'))

    optimizer = args.optimizer(PINN.parameters(), args.lr)

    phi = []
    power1 = np.array([10])
    power1 = torch.from_numpy(power1).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power1, 'lr': 0.001})
    phi.append(power1)

    power2 = np.array([15])
    power2 = torch.from_numpy(power2).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power2, 'lr': 0.001})
    phi.append(power2)

    power3 = np.array([10])
    power3 = torch.from_numpy(power3).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power3, 'lr': 0.001})
    phi.append(power3)

    power4 = np.array([15])
    power4 = torch.from_numpy(power4).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power4, 'lr': 0.001})
    phi.append(power4)

    power5 = np.array([10])
    power5 = torch.from_numpy(power5).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power5, 'lr': 0.001})
    phi.append(power5)

    power6 = np.array([10])
    power6 = torch.from_numpy(power6).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power6, 'lr': 0.001})
    phi.append(power6)

    power7 = np.array([15])
    power7 = torch.from_numpy(power7).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power7, 'lr': 0.001})
    phi.append(power7)

    power8 = np.array([15])
    power8 = torch.from_numpy(power8).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power8, 'lr': 0.001})
    phi.append(power8)

    power9 = np.array([15])
    power9 = torch.from_numpy(power9).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power9, 'lr': 0.001})
    phi.append(power9)

    power10 = np.array([10])
    power10 = torch.from_numpy(power10).float().cuda().requires_grad_(True)
    optimizer.add_param_group({'params': power10, 'lr': 0.001})
    phi.append(power10)

    # q_history = []
    loss_history = []
    loss_true = []

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

        # data
        u_pred_data = PINN(torch.cat([x_data, y_data], dim=1))
        mse_data = args.criterion(u_pred_data, u_data)
        # mse_data = criterionL1(u_pred_data, u_data)


        # true
        u_pred_true = PINN(torch.cat([x_true, y_true], dim=1))
        MAE_true = (1 / len(u_true)) * (sum(abs(u_pred_true - u_true)))
        loss_true.append(MAE_true.item())

        # loss
        loss = PDE_weight * mse_PDE + BC_weight * mse_BC + data_weight * mse_data
        loss_history.append([mse_PDE.item(), mse_BC.item(), mse_data.item(), loss.item()])

        # q_history.append([power1.item()])

        if epoch % 1000 == 0:
            print(
                'epoch:{:05d}, PDE: {:.08e}, BC: {:.08e},Data: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), mse_data.item(), loss.item()
                )
            )

        loss.backward()
        optimizer.step()

    torch.save(PINN.state_dict(), './result/{}/PINN_inverse_{}_{}.pth'.format(sample, Epoch, num))
    np.save('./result/{}/loss{}_{}.npy'.format(sample, Epoch, num), loss_history)
    np.save('./result/{}/MAE_loss{}.npy'.format(sample, Epoch, num), loss_true)



if __name__ == '__main__':
    setup_seed(0)
    parser = get_parser()
    args = parser.parse_args()
    PDE_weight = 1
    BC_weight = 1

    # num compare
    for i in range(12, 15):
        x_obs, y_obs, _ = obs_sample(n=i, n_b=4, uniform_point=True, random=False)
        sample = 'mesh'
        num = len(x_obs)
        data_weight = 1e4
        train(args, Epoch=5000, num=num, x_obs=x_obs, y_obs=y_obs, data_weight=data_weight, sample=sample)

    for i in [125, 148, 173]:
        print(i-1)
        x_obs, y_obs = sample_latin(N=i-1)
        sample = 'latin'
        # x_obs, y_obs = sample_uniform(N=i)
        # sample = 'uniform'
        num = i
        data_weight = 1e4
        train(args, Epoch=5000, num=num, x_obs=x_obs, y_obs=y_obs, data_weight=data_weight, sample=sample)

    for i in [125, 148, 173]:
        print(i-1)
        x_obs, y_obs = sample_uniform(N=i)
        sample = 'uniform'
        num = i
        data_weight = 1e4
        train(args, Epoch=5000, num=num, x_obs=x_obs, y_obs=y_obs, data_weight=data_weight, sample=sample)
