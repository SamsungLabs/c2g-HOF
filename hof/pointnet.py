from typing import NamedTuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetParameters(NamedTuple):
    """Parameters for PointNet.

    Attributes:
        point_code_dim: the hidden layer dimension.
        global_feature_dim: the global feature dimension.
        activation_function: the activation function on the output layer.
    """

    point_code_dim: int = 128
    global_feature_dim: int = 512
    activation_function: Callable = F.leaky_relu


class PointNet(nn.Module):
    """PointNet.

    Parameters:
        input_features: input points dimension.
        params: architectural parameters for the PointNet.
    """

    def __init__(
        self, input_features: int = 3, params: PointNetParameters = PointNetParameters()
    ):
        super().__init__()

        self.params = params

        self.c1 = nn.Conv1d(input_features, params.point_code_dim, 1)
        self.c2 = nn.Conv1d(params.point_code_dim, params.point_code_dim, 1)
        self.c3 = nn.Conv1d(params.point_code_dim, params.global_feature_dim, 1)

        self.c4 = nn.Conv1d(
            params.point_code_dim + params.global_feature_dim,
            params.global_feature_dim,
            1,
        )
        self.c5 = nn.Conv1d(params.global_feature_dim, params.global_feature_dim, 1)
        self.c6 = nn.Conv1d(params.global_feature_dim, params.global_feature_dim, 1)

    def forward(self, x: torch.tensor):
        assert x.dim() == 3
        x = x.transpose(-1, -2)

        out1 = self.params.activation_function(self.c1(x))
        out2 = self.params.activation_function(self.c2(out1))
        out3 = self.params.activation_function(self.c3(out2))

        global_code1 = out3.max(-1, keepdim=True).values.expand(-1, -1, x.shape[-1])
        # global_code1 = out3.max(-1).values

        out4 = self.params.activation_function(
            self.c4(torch.cat((out1, global_code1), 1))
        )
        out5 = self.params.activation_function(self.c5(out4))

        global_code2 = self.params.activation_function(self.c6(out5)).max(-1).values

        return global_code2


def test():
    batch, n_points, point_dim = 10, 10000, 3

    pn = PointNet(input_features=point_dim)

    input_points = torch.empty(batch, n_points, point_dim).normal_()

    output = pn(input_points)

    print(output.shape)


if __name__ == "__main__":
    test()
