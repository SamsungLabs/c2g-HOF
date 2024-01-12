from typing import Callable, List, NamedTuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPParams(NamedTuple):
    """Parameters for an MLP.

    Attributes:
        fc_sizes: a list of the number of units in each layer.
        fc_act_fn: the activation function for each hidden layer.
        out_act_fn: the activation function on the output layer.
    """

    fc_sizes: List[int]
    fc_act_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu
    out_act_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    use_layer_norm: bool = False


class MLP(nn.Module):
    """Multi-layer Perceptron.

    Parameters:
        in_shape: 1-d input dimension of the perceptron.
        out_shape: 1-d output dimension of the perceptron.
        params: architectural parameters for the MLP.
    """

    def __init__(
        self,
        in_shape: Union[int, Sequence[int]],
        out_shape: Union[int, Sequence[int]],
        params: MLPParams,
    ):
        super(MLP, self).__init__()

        # Store the parameters, converting int shapes to lists.
        self.in_shape = in_shape if isinstance(in_shape, Sequence) else [in_shape]
        self.out_shape = out_shape if isinstance(out_shape, Sequence) else [out_shape]
        self.params = params

        # Our MLP model should flatten out input/output dimensions for computation,
        # and then reshape the output at the very end.
        self.in_dim = np.product(in_shape)
        self.out_dim = np.product(out_shape)

        # Store activation funcs.
        self.fc_act_fn = params.fc_act_fn
        self.out_act_fn = params.out_act_fn

        # There are len(params.fc_sizes) + 2 layers in our network.
        self.fc_sizes = [self.in_dim] + params.fc_sizes + [self.out_dim]

        # Create layer lists.
        self.fc_layers = nn.ModuleList()
        if self.params.use_layer_norm:
            self.layer_norms = nn.ModuleList()

        # Add layers.
        for i in range(len(self.fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(self.fc_sizes[i], self.fc_sizes[i + 1]))

            # Optionally add layer norm.
            if self.params.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.fc_sizes[i + 1]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check if it's a batch. (hacky itertools because in_shape can be an int or a sequence.
        is_batch = len(input.size()) == len(self.in_shape) + 1

        # Reshape the input into a 1-D vector.
        if is_batch:
            x = input.reshape((-1, self.in_dim))
        else:
            x = input.reshape(self.in_dim)

        x = self.forward_first_n_layers(x, len(self.fc_sizes) - 2)

        # Connect the last layer. No layer norm because that keeps it from training.
        # TODO(beisner): Explain why this is the case.
        x = self.fc_layers[-1](x)
        x = self.out_act_fn(x)

        # Reshape the output back to the desired shape.
        if is_batch:
            return x.reshape((-1,) + tuple(self.out_shape))
        else:
            return x.reshape(self.out_shape)

    def forward_first_n_layers(self, input: torch.Tensor, first_n: int) -> torch.Tensor:

        # Anything up until the the last layer.
        assert first_n < len(self.fc_sizes) - 1

        x = input

        # Connect all the layers except for the last one.
        for i in range(first_n):
            x = self.fc_layers[i](x)

            # Optionally add layer norm.
            if self.params.use_layer_norm:
                x = self.layer_norms[i](x)

            x = self.fc_act_fn(x)

        return x
