import torch
from torch import nn
from torchvision.models import DenseNet201_Weights, densenet201


class CrossViewAttentionBlock(nn.Module):
    """Transformer encoder block for cross-view feature fusion."""

    def __init__(self, embed_dim, num_heads=8, dropout=0.2, mlp_ratio=4):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        q = k = v = self.layer_norm1(x)
        attn_output, _ = self.attention(q, k, v)
        x = x + attn_output
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CrossViewRegNet(nn.Module):
    """CrossViewRegNet for multi-view individual-tree species classification."""

    def __init__(
        self,
        n_classes,
        n_views=7,
        pretrained_backbone=True,
        attention_heads=8,
        attention_dropout=0.2,
        classifier_dropout=0.6,
    ):
        super().__init__()
        self.n_views = n_views

        weights = DenseNet201_Weights.DEFAULT if pretrained_backbone else None
        backbone = densenet201(weights=weights)
        z_dim = backbone.classifier.in_features
        self.z_dim = z_dim

        first_conv = backbone.features[0]
        if getattr(first_conv, "in_channels", None) == 3:
            new_weight = first_conv.weight.sum(dim=1, keepdim=True)
            new_conv = nn.Conv2d(
                1,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            new_conv.weight = nn.Parameter(new_weight)
            backbone.features[0] = new_conv

        backbone.classifier = nn.Identity()
        self.feature_extractor = backbone

        self.height_pathway = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

        self.attention_blocks = nn.Sequential(
            CrossViewAttentionBlock(
                embed_dim=z_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
            )
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, z_dim))
        self.classifier = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, images, heights):
        batch_size, n_views, channels, height, width = images.shape
        if n_views != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {n_views}.")

        view_inputs = images.reshape(batch_size * n_views, channels, height, width)
        image_features = self.feature_extractor(view_inputs)
        image_features = image_features.view(batch_size, n_views, self.z_dim)

        heights = heights.view(batch_size, -1).float()
        height_features = self.height_pathway(heights).unsqueeze(1)

        feature_sequence = torch.cat((image_features, height_features), dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        feature_sequence = torch.cat((cls_tokens, feature_sequence), dim=1)

        processed_sequence = self.attention_blocks(feature_sequence)
        cls_output = processed_sequence[:, 0]
        return self.classifier(cls_output)


SimpleView = CrossViewRegNet
