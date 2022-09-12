import argparse
import torch
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seq_net', default=[2, 50, 50, 50, 50, 1]
    )
    # parser.add_argument(
    #     '--seq_net', default=[2, 100, 100, 100, 100, 1]
    # )
    # parser.add_argument(
    #     '--seq_net', default=[2, 50, 50, 50, 1]
    # )
    # parser.add_argument(
    #     '--seq_net', default=[2, 50, 50, 50,50, 50, 1]
    # )
    # 训练信息
    parser.add_argument(
        '--epochs', default=20000, type=int
    )
    parser.add_argument(
        '--n_f', default=4000, type=int
    )
    parser.add_argument(
        '--n_f_1', default=1000, type=int
    )
    parser.add_argument(
        '--n_f_2', default=10000, type=int
    )
    parser.add_argument(
        '--n_b_l', default=5000, type=int
    )
    parser.add_argument(
        '--PDE_panelty', default=1.0, type=float
    )
    parser.add_argument(
        '--BC_panelty', default=1.0, type=float
    )
    parser.add_argument(
        '--lr', default=0.001, type=float
    )

    parser.add_argument(
        '--criterion', default=torch.nn.MSELoss()
    )
    parser.add_argument(
        '--optimizer', default=torch.optim.Adam
    )
    # 网络信息

    parser.add_argument(
        '--activation', default=torch.tanh
    )
    parser.add_argument(
        '--activ_name', default='tanh'
    )
    parser.add_argument(
        '--x_left', default=0., type=float
    )
    parser.add_argument(
        '--x_right', default=10, type=float
    )
    parser.add_argument(
        '--y_left', default=0., type=float
    )
    parser.add_argument(
        '--y_right', default=10, type=float
    )
    return parser
