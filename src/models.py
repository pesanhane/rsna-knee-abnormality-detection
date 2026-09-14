import torch
import torch.nn as nn


class MultiplanarKneeDINO(nn.Module):
    """Baseline multiplanar DINOv2 model with mean slice pooling."""

    def __init__(self, backbone, feature_dim=384, num_classes=12, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 3, num_classes),
        )

    def encode_plane(self, x):
        batch_size, num_slices, channels, height, width = x.shape
        x = x.reshape(
            batch_size * num_slices,
            channels,
            height,
            width,
        )
        features = self.backbone(x)
        features = features.reshape(batch_size, num_slices, -1)
        return features.mean(dim=1)

    def forward(self, sagittal, coronal, axial):
        sagittal_features = self.encode_plane(sagittal)
        coronal_features = self.encode_plane(coronal)
        axial_features = self.encode_plane(axial)

        fused = torch.cat(
            [sagittal_features, coronal_features, axial_features],
            dim=1,
        )
        return self.classifier(fused)
