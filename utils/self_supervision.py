from __future__ import annotations
from models.ResNet50_3D_foundation import *
from models.ResNet50_3D import *
from models.ViT_3DClassifier import *
import torch.nn.init as init
from collections.abc import Sequence
import torch
import torch.nn as nn
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock


class BarlowTwins(torch.nn.Module):
    def __init__(
            self,
            model,
            backbone_type,
            projector: tuple = (4096, 4096),
            scale_loss: float = 1.0 / 32,
            lambd: float = 5e-3,
    ):
        super().__init__()
        self.scale_loss = scale_loss
        self.lambd = lambd
        self.backbone = model
        self.backbone_type = backbone_type
        if self.backbone_type == "Resnet50":
            self.backbone.module.fc = nn.Identity()
            print(self.backbone)
            sizes = [2048] + list(projector)
        elif self.backbone_type == "Resnet_50_w":
            self.backbone.head = nn.Identity()
            sizes = [4096] + list(projector)
        elif self.backbone_type == "ViT":
            self.backbone.classification_head = nn.Identity()
            # Build projector
            sizes = [768] + list(projector)
        elif self.backbone_type == "Mamba":
            self.backbone.module.mlp = nn.Identity()
            # Build projector
            sizes = [448] + list(projector)
        elif self.backbone_type == "Swin":
            
            self.backbone.classifier = nn.Identity()  # .module
            sizes = [768] + list(projector)
            print(self.backbone)

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.DataParallel(nn.Sequential(*layers))
        self.bn = nn.DataParallel(nn.BatchNorm1d(sizes[-1], affine=False))

        # Initialize the weights and biases of the projector
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)  # Initialize bias to zero
        print(self.projector)
        self.backbone.train()
        self.projector.train()
        self.bn.train()
        
    def forward(self, y1, y2):

        if self.backbone_type == "ViT":
            y1, _ = self.backbone(y1)
            y2, _ = self.backbone(y2)
            y1 = y1[:, 0, :]  # Prendo solo il class token
            y2 = y2[:, 0, :]  # Prendo solo il class token
        else:
            y1 = self.backbone(y1)
            y2 = self.backbone(y2)


        self.projector.train()
        z1 = self.projector(y1)
        z2 = self.projector(y2)

        c = self.bn(z1).T @ self.bn(z2)

        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    # Initialize the model with necessary parameters
    def __init__(
            self,
            in_channels: int,
            img_size: Sequence[int] | int,
            patch_size: Sequence[int] | int,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_layers: int = 12,
            num_heads: int = 12,
            pos_embed: str = "conv",
            classification: bool = False,
            num_classes: int = 2,
            dropout_rate: float = 0.0,
            spatial_dims: int = 3,
            post_activation="Tanh",
            qkv_bias: bool = False,
            save_attn: bool = False,
            pretrain: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()
        self.pretrain = pretrain
        # Check if dropout_rate is between 0 and 1
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        # Check if hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification

        # Define the patch embedding block
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # Define the transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )

        # Define the layer normalization
        self.norm = nn.LayerNorm(hidden_size)

        # Define the classification head if classification is True
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                 self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                 self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    # Define the forward method
    # The input is passed through the patch embedding block and then through each transformer block
    # The output is then normalized
    def forward(self, x):
        x = self.patch_embedding(x)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        if hasattr(self, "classification_head") and self.pretrain==False:
            x = self.classification_head(x[:, 0])

        return x, hidden_states_out


if __name__ == "__main__":
    """model = ResNet(Bottleneck, [3, 4, 6, 3], n_classes=1, pretrained=False, sshead=False)
    bt = BarlowTwins(model, backbone_type="Resnet_50", projector=(2048, 2048))
    emb1 = torch.rand(2, 1, 282, 180, 180)
    emb2 = torch.rand(2, 1, 282, 180, 180)
    print(bt(emb1, emb2))"""

    model = ViT(
            in_channels=1,
            img_size=(288, 192, 192),
            patch_size=(48, 48, 48),
            pos_embed="conv",  # "perceptron"
            classification=True,
            num_classes=1,
            pretrain=True,
        )
    bt = BarlowTwins(model, backbone_type="ViT", projector=(768, 768))

    emb1 = torch.rand(2, 1, 288, 192, 192)
    emb2 = torch.rand(2, 1, 288, 192, 192)
    print(bt(emb1, emb2))


