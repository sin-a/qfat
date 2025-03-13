import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


class IdentityEncoder(nn.Module):
    def __init__(self):
        """
        Identity encoder that returns the input as-is while ensuring the output
        shape is (B, T, dim_size).
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure the input is of shape (B, T, dim_size).

        Args:
            x (torch.Tensor): Input tensor. Thape can be:
                              - (dim_size,)
                              - (B, dim_size)
                              - (B, T, dim_size)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, dim_size).
        """
        if x.ndim == 1:  # Single feature vector (dim_size)
            x = x.unsqueeze(
                (0, 1)
            )  # Add batch and sequence dimensions -> (1, 1, dim_size)
        elif x.ndim == 2:  # Sequence of feature vectors (T, dim_size)
            x = x.unsqueeze(0)  # Add batch dimension -> (B, T, dim_size)
        elif x.ndim != 3:  # Ensure valid input shape
            raise ValueError(
                "Input must have shape (dim_size), (B, dim_size), or (B, T, dim_size)"
            )

        return x


class HierarchicalResNet(nn.Module):
    def __init__(self, freeze_weights: bool = True):
        """
        ResNet-based hierarchical encoder for image embeddings.

        Args:
            freeze_weights (bool): Whether to freeze ResNet weights.
        """
        super().__init__()
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.freeze_weights = freeze_weights

        # Extract intermediate layers
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # Until layer1
        self.layer2 = nn.Sequential(*list(resnet.children())[5])  # layer2
        self.layer3 = nn.Sequential(*list(resnet.children())[6])  # layer3
        self.layer4 = nn.Sequential(
            *list(resnet.children())[7]
        )  # layer4 (final convolution)

        if freeze_weights:
            for param in resnet.parameters():
                param.requires_grad = False
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()

        # Define a head to reduce dimensionality after concatenation
        self.head = nn.Sequential(
            nn.Linear(64 + 128 + 256 + 512, 1024),  # Combined channels
            nn.ReLU(),
        )

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the input tensor for ResNet.

        Args:
            img (torch.Tensor): Input image tensor. Shape can be:
                                - (H, W, C)
                                - (T, H, W, C)
                                - (B, T, H, W, C)

        Returns:
            torch.Tensor: Preprocessed tensor of shape (B*T, C, 224, 224).
        """
        if img.ndim == 3:  # (H, W, C)
            img = img.unsqueeze(0).unsqueeze(1)
        elif img.ndim == 4:  # (T, H, W, C)
            img = img.unsqueeze(0)

        if img.ndim != 5:
            raise ValueError(
                "Input must have shape (H, W, C), (T, H, W, C), or (B, T, H, W, C)"
            )

        img = einops.rearrange(
            img, "b t h w c -> (b t) c h w"
        )  # flatten sequence into batch dimension
        img = img / 255.0
        if img.shape[-2:] != (224, 224):
            img = F.interpolate(img, size=224, mode="bilinear", align_corners=False)
        img_norm = self.transform(img)
        return img_norm

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute ResNet hierarchical embeddings for the input images.

        Args:
            img (torch.Tensor): Input tensor. Shape can be:
                                - (H, W, C)
                                - (B, H, W, C)
                                - (B, T, H, W, C)

        Returns:
            torch.Tensor: ResNet embeddings of shape (B, T, 512).
        """
        img_preprocessed = self.preprocess(img)

        # Extract intermediate features
        x1 = self.layer1(img_preprocessed)  # (B*T, 64, H1, W1)
        x2 = self.layer2(x1)  # (B*T, 128, H2, W2)
        x3 = self.layer3(x2)  # (B*T, 256, H3, W3)
        x4 = self.layer4(x3)  # (B*T, 512, H4, W4)

        # Perform global pooling and flatten
        x1 = nn.functional.adaptive_avg_pool2d(x1, (1, 1)).flatten(1)
        x2 = nn.functional.adaptive_avg_pool2d(x2, (1, 1)).flatten(1)
        x3 = nn.functional.adaptive_avg_pool2d(x3, (1, 1)).flatten(1)
        x4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1)).flatten(1)

        # Concatenate all features
        combined_features = torch.cat([x1, x2, x3, x4], dim=1)  # (B*T, 960)

        # Reduce dimensionality
        embeddings = self.head(combined_features)  # (B*T, 512)

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        batch_size, seq_len = img.shape[0], img.shape[1] if img.ndim == 5 else 1
        embeddings = einops.rearrange(
            embeddings, "(b t) d -> b t d", b=batch_size, t=seq_len
        )
        return embeddings
