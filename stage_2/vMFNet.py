import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from config import config
from helper import getVmfKernels

cfg = config()


class Conv1o1Layer(nn.Module):
    def __init__(self, weights):
        super(Conv1o1Layer, self).__init__()
        self.weight = nn.Parameter(weights)

    def forward(self, x):
        weight = self.weight
        xnorm = torch.norm(x, dim=1, keepdim=True)
        boo_zero = (xnorm == 0).float()
        xnorm = xnorm + boo_zero
        xn = x / xnorm
        wnorm = torch.norm(weight, dim=1, keepdim=True)
        weightnorm2 = weight / wnorm
        out = F.conv2d(xn, weightnorm2)
        if torch.sum(torch.isnan(out)) > 0:
            print('isnan conv1o1')
        return out


class ActivationLayer(nn.Module):
    """Compute activation of a Tensor. The activation could be a exponent or a
    binary thresholding.
    """

    def __init__(self, vMF_kappa, threshold=0.0):
        super(ActivationLayer, self).__init__()
        self.vMF_kappa = vMF_kappa
        self.threshold = threshold

    def forward(self, x):
        x = torch.exp(self.vMF_kappa * x) * (x > self.threshold).type_as(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_downconv = 3, in_chn = 3):
        super().__init__()

        # a tunable number of DownConv blocks in the architecture
        self.n_downconv = n_downconv

        layer_list = [ # The two mandatory initial layers
            nn.Conv2d(in_channels=in_chn, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        ]
        # 'n_downconv' number of DownConv layers (In the CVPR paper, it was 3)
        for i in range(self.n_downconv):
            layer_list.extend([
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), 
                    nn.ReLU(),
                    
                ])

        layer_list.append( # The one mandatory end layer
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
        )

        # register the Sequential module
        self.encoder = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward pass; a final clamping is applied
        return torch.clamp(self.encoder(x), 0, 1)


class Decoder(nn.Module):
    def __init__(self, n_upconv = 3, out_chn = 3):
        super().__init__()

        # a tunable number of DownConv blocks in the architecture
        self.n_upconv = n_upconv

        layer_list = [ # The one mandatory initial layers
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        ]
        # 'n_upconv' number of UpConv layers (In the CVPR paper, it was 3)
        for i in range(self.n_upconv):
            layer_list.extend([
                    nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1), 
                    nn.ReLU(),
                    nn.PixelShuffle(2),
                ])
        # The mandatory final layer
        layer_list.extend([
                nn.Conv2d(in_channels=64, out_channels=out_chn*4, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2)
            ])

        # register the Sequential module
        self.decoder = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward pass; a final clamping is applied
        return torch.clamp(self.decoder(x), 0, 1)


class CompCSD(nn.Module):
    def __init__(self, train_flag=True):
        super(CompCSD, self).__init__()

        self.activation_layer = ActivationLayer(cfg.vMF_kappa)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.train_flag = train_flag

    def forward(self, x):
        features = self.encoder(x)
        vc_activations = self.conv1o1(features) # fp*uk
        vmf_activations = self.activation_layer(vc_activations) # L
        norm_vmf_activations = torch.zeros_like(vmf_activations)
        norm_vmf_activations = norm_vmf_activations.to(cfg.device)
        for i in range(vmf_activations.size(0)):
            norm_vmf_activations[i, :, :, :] = F.normalize(vmf_activations[i, :, :, :], p=1, dim=0)
        self.vmf_activations = norm_vmf_activations
        self.vc_activations = vc_activations
        decoding_features = self.compose(norm_vmf_activations)
        Ih = self.decoder(decoding_features)

        if self.train_flag:
            return Ih, features
        else:
            return Ih, features, norm_vmf_activations, decoding_features
 
    def load_vmf_kernels(self, dict_dir):
        weights = getVmfKernels(dict_dir)
        self.conv1o1 = Conv1o1Layer(weights).to(cfg.device)

    def compose(self, vmf_activations):
        kernels = self.conv1o1.weight
        kernels = kernels.squeeze(2).squeeze(2)

        features = torch.zeros([vmf_activations.size(0), kernels.size(1), vmf_activations.size(2), vmf_activations.size(3)])
        features = features.to(cfg.device)

        for k in range(vmf_activations.size(0)):
            single_vmf_activations = vmf_activations[k]
            single_vmf_activations = torch.permute(single_vmf_activations, (1, 2, 0))  # [512, 72, 72]
            feature = torch.matmul(single_vmf_activations, kernels)
            feature = torch.permute(feature, (2, 0, 1))
            features[k, :, :, :] = feature
        return features


def initialize_weights(model, init="xavier"):
    init_func = None
    if init == "xavier":
        init_func = torch.nn.init.xavier_normal_
    elif init == "kaiming":
        init_func = torch.nn.init.kaiming_normal_
    elif init == "gaussian" or init == "normal":
        init_func = torch.nn.init.normal_

    if init_func is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    else:
        print(
            "Error when initializing model's weights, {} either doesn't exist or is not a valid initialization function.".format(
                init), \
            file=sys.stderr)
