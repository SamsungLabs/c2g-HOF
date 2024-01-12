import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal, Independent
from hof.mlp import MLP, MLPParams


class RBFNet(nn.Module):
    """Radial Basis Function Network.

    Parameters:
        in_mlp: MLP input dimension.
        out_mlp: MLP output dimension.
         kernels: kernel size
         hidden_neurons: hidden neuron size
         net_out_dim: network output dimension
    """

    @staticmethod
    def parameter_count_for_kernels(kernels, dim=3):
        return (dim + dim ** 2) * kernels

    def __init__(
        self,
        in_mlp: int,
        out_mlp: int,
        kernels: int = 32,
        hidden_neurons: int = 512,
        net_out_dim: int = 3,
    ):
        super(RBFNet, self).__init__()
        self.params = MLPParams([in_mlp, out_mlp])
        self.mlp = MLP(in_mlp, out_mlp, self.params)

        self.fc1 = nn.Linear(kernels, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc4 = nn.Linear(hidden_neurons, hidden_neurons)
        # self.fc5 = nn.Linear(hidden_neurons, 3)    # generative model
        self.fc5 = nn.Linear(hidden_neurons, net_out_dim)  # cost2go or claasification

        self.kernels = kernels

    def unpack_theta(self, rbf_theta, point_dim):
        mus = rbf_theta[: self.kernels * point_dim].view(self.kernels, point_dim)
        sigmas = rbf_theta[self.kernels * point_dim :].view(
            self.kernels, point_dim, point_dim
        ) / 10 + torch.eye(point_dim, device=rbf_theta.device).unsqueeze(0).repeat(
            self.kernels, 1, 1
        )
        sigmas = sigmas @ sigmas.permute(0, 2, 1)

        return mus, sigmas

    def forward(self, points, rbf_theta):
        assert rbf_theta.dim() == 1
        assert points.dim() == 2
        rbf_theta = self.mlp(rbf_theta)
        mus, sigmas = self.unpack_theta(
            rbf_theta, points.shape[1]
        )  # mus : [kernel, dim], sigmas = [kernel,dim, dim]

        distributions = MultivariateNormal(mus, sigmas)
        xyz_duped = points.unsqueeze(1).repeat(
            1, self.kernels, 1
        )  # [Points, Kernels, Dim] # duplicated as many as kernels
        rbf_features = distributions.log_prob(xyz_duped)  # [Points, kernel]

        fc1_out = self.fc1(rbf_features).relu()  # [Points, 126]
        fc2_out = self.fc2(fc1_out).relu()  # [Points, 126]
        fc3_out = self.fc3(fc2_out).relu()  # [Points, 126]
        fc4_out = self.fc4(fc3_out).relu()
        xyz_hat = self.fc5(fc4_out)  # [Points, 126]

        return xyz_hat


class RBFDiagNet(nn.Module):

    """Radial Basis Function Network with a diagonal covariance matrix.

    Parameters:
        in_mlp: MLP input dimension.
        out_mlp: MLP output dimension.
         kernels: kernel size
         hidden_neurons: hidden neuron size
         net_out_dim: network output dimension
    """

    @staticmethod
    def parameter_count_for_kernels(kernels, dim=3):
        return (dim + dim) * kernels

    def __init__(
        self,
        in_mlp: int,
        out_mlp: int,
        kernels: int = 32,
        hidden_neurons: int = 512,
        net_out_dim: int = 3,
    ):
        super(RBFDiagNet, self).__init__()
        self.params = MLPParams([in_mlp, out_mlp])
        self.mlp = MLP(in_mlp, out_mlp, self.params)

        self.fc1 = nn.Linear(kernels, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc4 = nn.Linear(hidden_neurons, hidden_neurons)
        # self.fc5 = nn.Linear(hidden_neurons, 3)    # generative model
        self.fc5 = nn.Linear(hidden_neurons, net_out_dim)  # cost2go or claasification

        self.kernels = kernels

    def unpack_theta(self, rbf_theta, point_dim):
        mus = rbf_theta[: self.kernels * point_dim].view(self.kernels, point_dim)
        scales = (
            rbf_theta[self.kernels * point_dim :].view(self.kernels, point_dim) / 10 + 1
        )
        return mus, scales

    def forward(self, points, rbf_theta):
        assert rbf_theta.dim() == 1
        assert points.dim() == 2
        # assert points.shape[1] == 3
        rbf_theta = self.mlp(rbf_theta)
        mus, scales = self.unpack_theta(
            rbf_theta, points.shape[1]
        )  # mus : [kernel, dim], sigmas = [kernel,dim, dim]

        distributions = Normal(mus, scales)
        diagn = Independent(distributions, 1)
        xyz_duped = points.unsqueeze(1).repeat(
            1, self.kernels, 1
        )  # [Points, Kernels, Dim] # duplicated as many as kernels
        rbf_features = diagn.log_prob(xyz_duped)  # [Points, kernel]

        fc1_out = self.fc1(rbf_features).relu()  # [Points, 126]
        fc2_out = self.fc2(fc1_out).relu()  # [Points, 126]
        fc3_out = self.fc3(fc2_out).relu()  # [Points, 126]
        fc4_out = self.fc4(fc3_out).relu()
        xyz_hat = self.fc5(fc4_out)  # [Points, 126]

        return xyz_hat


def build_HOF_rbf(
    args,
    device,
    point_dim: int,
    output_dim: int = 1,
):

    """build radial basis function network.

    Parameters:
        args: parameters
        device: a torch.device for the network
        point_dim: input point dimension
        output_dim: the network output dimension
    """

    output_dims = [RBFNet.parameter_count_for_kernels(kernels=args.rbf, dim=point_dim)]
    rbfnet = RBFNet(
        in_mlp=args.conv_output_dim,
        out_mlp=output_dims[0],
        kernels=args.rbf,
        hidden_neurons=args.width,
        net_out_dim=output_dim,
    ).to(device)

    return rbfnet, output_dims


def build_HOF_Diag_rbf(
    args,
    device,
    point_dim: int,
    output_dim: int = 1,
):

    """build diagonal radial basis function network.

    Parameters:
        args: parameters
        device: a device for the network
        point_dim: input point dimension
        output_dim: the network output dimension
    """
    if args.rbf_type == "diagonal":
        output_dims = [
            RBFDiagNet.parameter_count_for_kernels(kernels=args.rbf, dim=point_dim)
        ]
        rbfnet = RBFDiagNet(
            in_mlp=args.conv_output_dim,
            out_mlp=output_dims[0],
            kernels=args.rbf,
            hidden_neurons=args.width,
            net_out_dim=output_dim,
        ).to(device)
    else:
        output_dims = [
            RBFNet.parameter_count_for_kernels(kernels=args.rbf, dim=point_dim)
        ]
        rbfnet = RBFNet(
            in_mlp=args.conv_output_dim,
            out_mlp=output_dims[0],
            kernels=args.rbf,
            hidden_neurons=args.width,
            net_out_dim=output_dim,
        ).to(device)

    return rbfnet, output_dims


def test():

    batch, n_points, point_dim = 10, 10000, 2
    kernel_size = 32
    output_dims = [
        RBFNet.parameter_count_for_kernels(kernels=kernel_size, dim=point_dim)
    ]

    rbfnet = RBFNet(
        in_mlp=256,
        out_mlp=output_dims[0],
        kernels=kernel_size,
        hidden_neurons=126,
        net_out_dim=1,
    )
    mapping_inputs = torch.empty(n_points, point_dim).uniform_(-1, 1) * 10
    theta1 = torch.empty(256).uniform_(-1, 1)
    object_prediction = rbfnet(mapping_inputs, theta1)  # 1000 X 3
    print(object_prediction.shape)


if __name__ == "__main__":
    test()
