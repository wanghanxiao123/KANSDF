# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# This code is the implementation of the DiGS model and loss functions
# It was partly based on SIREN and SAL implementation and architecture but with several significant modifications.
# for the original SIREN version see: https://github.com/vsitzmann/siren
# for the original SAL version see: https://github.com/matanatz/SAL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import utils.utils as utils
from models.torchmeta.modules.module import MetaModule
from models.torchmeta.modules.container import MetaSequential
from models.torchmeta.modules.utils import get_subdict
from collections import OrderedDict
from models.losses import DiGSLoss
import torch
import torch.nn as nn

class ChebyKAN(nn.Module):
    def __init__(self, in_features, out_features, degree):
        super(ChebyKAN, self).__init__()
        assert degree > 0, "Degree of the Chebyshev polynomials must be greater than 0"
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        std = 1 / (self.in_features * (self.degree + 1))
        self.coefficients = nn.Parameter(torch.randn(in_features, out_features, self.degree + 1) * std)

    def forward(self, x):
        # x: (batch_size, in_features)
        # normalize x between -1 and 1
        x = torch.tanh(x)
        # Initialize cheby_values with shape (batch_size, in_features, degree+1)
        cheby_values = torch.ones(x.size(0), self.in_features, self.degree + 1, device=x.device)
        cheby_values[:, :, 1] = x

        for i in range(2, self.degree + 1):
            # Compute next_value using values from cheby_values without in-place operations
            next_value = 2 * x * cheby_values[:, :, i - 1] - cheby_values[:, :, i - 2]
            cheby_values = cheby_values.clone()  # Cloning to ensure no in-place modification
            cheby_values[:, :, i] = next_value

        # cheby_values: (batch_size, in_features, degree+1)
        # multiply by coefficients (in_features, out_features, degree+1)
        return torch.einsum('bid,ijd->bj', cheby_values, self.coefficients)

# Example Usage
# model = ChebyKAN(in_features=128, out_features=64, degree=3)
# input_tensor = torch.randn(10, 128)  # (batch_size, in_features)
# output = model(input_tensor)
# print(output.shape)  # Expected shape: (10, 64)

class Decoder(nn.Module):

    def forward(self, *args, **kwargs):
        return self.fc_block(*args, **kwargs)

class DiGSNetwork(nn.Module):
    def __init__(self, latent_size, in_dim=3, decoder_hidden_dim=256, nl='sine', encoder_type=None,
                 decoder_n_hidden_layers=8, init_type='siren', sphere_init_params=[1.6, 1.0]):
        super().__init__()
        self.encoder_type = encoder_type
        self.init_type = init_type
        if encoder_type == 'autodecoder':
            # latent_size will stay as input latent size
            pass
        elif encoder_type == 'none':
            latent_size = 0
        else:
            raise ValueError("unsupported encoder type")
        self.decoder = Decoder()
        self.decoder.fc_block = FCBlock(in_dim + latent_size, 1, num_hidden_layers=decoder_n_hidden_layers, hidden_features=decoder_hidden_dim,
                                outermost_linear=True, nonlinearity=nl, init_type=init_type,
                                sphere_init_params=sphere_init_params)  # SIREN decoder

    def forward(self, non_mnfld_pnts, mnfld_pnts=None):
        # shape is (bs, npoints, in_dim+latent_size) for both inputs, npoints could be different sizes
        batch_size = non_mnfld_pnts.shape[0]
        if not mnfld_pnts is None and self.encoder_type == 'autodecoder':
            # Assume inputs have latent vector concatted with [xyz, latent]
            latent = non_mnfld_pnts[:,:,3:]
            latent_reg = latent.norm(dim=-1).mean()
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
        elif mnfld_pnts is not None:
            manifold_pnts_pred = self.decoder(mnfld_pnts.view(-1, mnfld_pnts.shape[-1])).reshape(batch_size, -1)
            latent = None
            latent_reg = None
        else:
            manifold_pnts_pred = None
            latent = None
            latent_reg = None

        # Off manifold points
        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts.view(-1, non_mnfld_pnts.shape[-1])).reshape(batch_size, -1)

        return {"manifold_pnts_pred": manifold_pnts_pred,
                "nonmanifold_pnts_pred": nonmanifold_pnts_pred,
                "latent_reg": latent_reg,
                "latent": latent}


class FCBlock0(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', init_type='siren',
                 sphere_init_params=[1.6,1.0]):
        super().__init__()
        print("decoder initialising with {} and {}".format(nonlinearity, init_type))

        self.first_layer_init = None
        self.sphere_init_params = sphere_init_params
        self.init_type = init_type

        nl_dict = {'sine': Sine(), 'relu': nn.ReLU(inplace=True), 'softplus': nn.Softplus(beta=100),
                    'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}
        nl = nl_dict[nonlinearity]

        self.net = []
        self.net.append(MetaSequential(BatchLinear(in_features, hidden_features), nl))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(BatchLinear(hidden_features, hidden_features), nl))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features), nl))

        self.net = MetaSequential(*self.net)

        if init_type == 'siren':
            self.net.apply(sine_init)
            self.net[0].apply(first_layer_sine_init)

        elif init_type == 'geometric_sine':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_geom_sine_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)

        elif init_type == 'mfgi':
            self.net.apply(geom_sine_init)
            self.net[0].apply(first_layer_mfgi_init)
            self.net[1].apply(second_layer_mfgi_init)
            self.net[-2].apply(second_last_layer_geom_sine_init)
            self.net[-1].apply(last_layer_geom_sine_init)

        elif init_type == 'geometric_relu':
            self.net.apply(geom_relu_init)
            self.net[-1].apply(geom_relu_last_layers_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))

        if self.init_type == 'mfgi' or self.init_type == 'geometric_sine':
            radius, scaling = self.sphere_init_params
            output = torch.sign(output)*torch.sqrt(output.abs()+1e-8)
            output -= radius # 1.6
            output *= scaling # 1.0

        return output

class FCBlock(MetaModule):
    '''A fully connected neural network designed to exclusively utilize the ChebyKAN layer for all processing,
    making it highly specialized for tasks where Chebyshev polynomials are advantageous.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, degree=8,outermost_linear=False, nonlinearity='sine', init_type='siren',
                 sphere_init_params=[1.6,1.0]):
        # in_features, out_features, num_hidden_layers, hidden_features,
        #          outermost_linear=False, nonlinearity='sine', init_type='siren',
        #          sphere_init_params=[1.6,1.0]):
    # def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
    #              outermost_linear=False, nonlinearity='sine', init_type='siren',
    #              sphere_init_params=[1.6,1.0]):
        super().__init__()
        print("Initialising with exclusive use of ChebyKAN layers")

        # Define the sequence of ChebyKAN layers
        self.net = []
        # Input to the first ChebyKAN layer
        self.net.append(ChebyKAN(in_features, hidden_features, degree))

        # Intermediate ChebyKAN layers
        for _ in range(1, num_hidden_layers - 1):
            self.net.append(ChebyKAN(hidden_features, hidden_features, degree))

        # Output ChebyKAN layer
        self.net.append(ChebyKAN(hidden_features, out_features, degree))

        self.net = MetaSequential(*self.net)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output
class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def forward(self, input):
        # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


################################# SIREN's initialization ###################################
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

################################# sine geometric initialization ###################################

def geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30

def first_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
            m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
            m.weight.data /= 30
            m.bias.data /= 30


def second_last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_output = m.weight.size(0)
            assert m.weight.shape == (num_output, num_output)
            m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
            m.bias.data = 0.5 * np.pi * torch.ones(num_output, ) + 0.001 * torch.randn(num_output)
            m.weight.data /= 30
            m.bias.data /= 30

def last_layer_geom_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (1, num_input)
            assert m.bias.shape == (1,)
            # m.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
            m.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
            m.bias.data = torch.zeros(1) + num_input


################################# multi frequency geometric initialization ###################################
periods = [1, 30] # Number of periods of sine the values of each section of the output vector should hit
# periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
portion_per_period = np.array([0.25, 0.75]) # Portion of values per section/period

def first_layer_mfgi_init(m):
    global periods
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            num_output = m.weight.size(0)
            num_per_period = (portion_per_period * num_output).astype(int) # Number of values per section/period
            assert len(periods) == len(num_per_period)
            assert sum(num_per_period) == num_output
            weights = []
            for i in range(0, len(periods)):
                period = periods[i]
                num = num_per_period[i]
                scale = 30/period
                weights.append(torch.zeros(num,num_input).uniform_(-np.sqrt(3 / num_input) / scale, np.sqrt(3 / num_input) / scale))
            W0_new = torch.cat(weights, axis=0)
            m.weight.data = W0_new

def second_layer_mfgi_init(m):
    global portion_per_period
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            assert m.weight.shape == (num_input, num_input)
            num_per_period = (portion_per_period * num_input).astype(int) # Number of values per section/period
            k = num_per_period[0] # the portion that only hits the first period
            # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
            W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.0005
            W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
            W1_new[:k, :k] = W1_new_1
            m.weight.data = W1_new

################################# geometric initialization used in SAL and IGR ###################################
def geom_relu_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            out_dims = m.out_features

            m.weight.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(out_dims))
            m.bias.data = torch.zeros_like(m.bias.data)

def geom_relu_last_layers_init(m):
    radius_init = 1
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.normal_(mean=np.sqrt(np.pi) / np.sqrt(num_input), std=0.00001)
            m.bias.data = torch.Tensor([-radius_init])