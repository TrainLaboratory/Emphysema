import os
from pathlib import Path
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETREncoder(nn.Module):
    def __init__(self, num_classes=2, sshead=False):
        super().__init__()
        self.sshead = sshead

        # Initialize Swin UNETR Encoder (without the decoder)
        self.encoder = SwinUNETR(
            img_size=(192, 192, 192),
            in_channels=1,
            out_channels=14,  # This is unused in classification mode
            feature_size=48,
            use_checkpoint=True,
        ).swinViT  # Use only the transformer encoder

        # Apply weight initialization
        self.encoder.apply(self.initialize_weights)


        # Global Adaptive Pooling (to reduce spatial dimensions)
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Fully connected classification layer
        self.classifier = nn.Linear(768, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # Self-supervised auxiliary head (optional)
        if self.sshead:
            print("Creating ssHead...")
            self.ss_head = nn.Linear(768, 4)
            nn.init.xavier_uniform_(self.ss_head.weight)
            nn.init.zeros_(self.ss_head.bias)

    def forward(self, x, return_embeddings=False):
        # Extract features from SwinUNETR encoder
        hidden_states = self.encoder(x)

        # Use the final layer of the encoder
        x = self.pool(hidden_states[-1])  # Taking the last feature map
        
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 768)
        
        if return_embeddings:
        	return x
        	
        # Classification output
        out = self.classifier(x)

        if self.sshead:
            ssh = self.ss_head(x)  # Self-supervised head output
            return out, ssh
        return out

    def initialize_weights(self, m):
        """Initializes the weights using appropriate distributions."""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)