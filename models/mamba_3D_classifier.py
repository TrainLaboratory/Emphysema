import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_library.mamba_ssm import Mamba
import torch.nn.init as init


# x installare mamba_ssm (1.0.1) installare prima causal-conv1d (1.2.1)


def initialize_weights(module):
    if isinstance(module, nn.Conv3d):
        print("Initializing Conv3D...")
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm3d):
        print("Initializing BatchNorm3D...")
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        print("Initializing Linear...")
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,  # input channels
        out_planes,  # output channels
        kernel_size=3,  # Size of the convolutional kernel
        stride=stride,  # Stride of the convolution
        padding=dilation,  # Padding added to all sides of the input
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def feature_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class Se(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.norm = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, channels = x.shape[:2]
        x = self.nin(x)
        x = self.norm(x)
        x = self.relu(x)
        residual = x
        assert channels == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(batch, channels, n_tokens).transpose(-1, -2)
        return x_flat, residual, img_dims


class Sd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.nin = conv1x1(dim, dim)
        self.norm = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual, batch, channels, img_dims):
        out = x.transpose(-1, -2).reshape(batch, channels, *img_dims)
        out += residual
        out = self.nin2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out


class Mamba_Block(nn.Module):

    def __init__(self, dim, d_state=8, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        x_mamba = self.mamba(x)
        return x_mamba


class HierarchicalFeatureExtractor(nn.Module):
    def __init__(self, channels, blocks, pooling_size=(1, 1, 1)):
        super(HierarchicalFeatureExtractor, self).__init__()

        self.layer1 = feature_layer(channels, channels * 2, blocks, stride=2)
        self.layer2 = feature_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = feature_layer(channels * 4, channels * 8, blocks, stride=2)

        self.pooling = nn.AdaptiveAvgPool3d(pooling_size)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        pooled_x1 = self.pooling(x1)
        pooled_x2 = self.pooling(x2)
        pooled_x3 = self.pooling(x3)

        return pooled_x1, pooled_x2, pooled_x3


class FeatureExtractor(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class ClassificationHead(nn.Module):
    def __init__(self, channels, number_classes):
        super(ClassificationHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 14, channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(channels, number_classes)
        )

    def forward(self, x):
        return self.mlp(x)

# Hybrid approach CNN-MAMBA for emphysema classification
class Cnn_Mamba(nn.Module):

    def __init__(self, in_ch=1, channels=32, blocks=3, number_classes=1):
        super(Cnn_Mamba, self).__init__()

        self.E = FeatureExtractor(in_ch, channels, stride=2, kernel_size=3)  # low-level feature extractor
        self.S_encoder = Se(dim=channels)  # encoder 1x1
        self.mamba_block_1 = Mamba_Block(dim=channels, d_state=8, d_conv=4, expand=2)  # Mamba block
        self.S_decoder = Sd(channels)  # decoder 1x1
        self.E_f = HierarchicalFeatureExtractor(channels, blocks)  # hierarchical feature extractor
        self.mamba_block_2 = Mamba_Block(dim=channels * 2, d_state=8, d_conv=2, expand=2)  # Mamba block
        self.classification_head = ClassificationHead(channels, number_classes)  # classification head

        self.apply(initialize_weights)

    def forward(self, x, return_embeddings=False):
        Z = self.E(x)
        Z_tilde = self.S_encoder(Z)
        Z_tilde_s = self.mamba_block_1(Z_tilde)
        Z_s = self.S_decoder(Z_tilde_s)

        Z_s_r = Z_s + Z
        pooled_f_1, pooled_f_2, pooled_f_3 = self.E_f(Z_s_r)

        f_concat = torch.cat(
            (pooled_f_1.reshape(Z.shape[0], Z.shape[1] * 2, 1),
             pooled_f_2.reshape(Z.shape[0], Z.shape[1] * 2, 2),
             pooled_f_3.reshape(Z.shape[0], Z.shape[1] * 2, 4)),
            dim=2)

        F_s = self.mamba_block_2(f_concat)

        F_s_r = F_s + f_concat
        F_s_r_flattened = F_s_r.reshape(Z.shape[0], -1)

        if return_embeddings:
            return F_s_r_flattened

        return self.classification_head(F_s_r_flattened)


if __name__ == "__main__":
    model = Cnn_Mamba().cuda()
    print(model)
    input = torch.ones((8, 1, 288, 192, 192)).cuda()  # default: (8, 1, 128, 128, 128)
    output = model(input)
    print(output, output.shape)
