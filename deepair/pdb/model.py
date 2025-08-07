import torch
import torch.nn as nn
# If an SE(3) Transformer library is available, import it
try:
    from se3_transformer_pytorch import SE3Transformer
except ImportError:
    SE3Transformer = None


class SimplePointNet(nn.Module):
    """A simple pointwise MLP that processes each point independently and pools globally."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super(SimplePointNet, self).__init__()
        self.input_dim = input_dim  # dimension of per-point features (coords + other features)
        self.hidden_dim = hidden_dim
        # We will embed coordinates + input features together
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)  # output per-point embedding
        )

    def forward(self, coords, features, mask=None):
        # Concatenate coords and features for each atom
        # coords: (B, N, 3), features: (B, N, F_feat)
        x = torch.cat([coords, features], dim=-1)  # shape (B, N, 3 + F_feat)
        # Flatten points to apply MLP
        B, N, _ = x.shape
        x_flat = x.view(B * N, -1)
        point_emb = self.mlp(x_flat)            # (B*N, hidden_dim)
        point_emb = point_emb.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)
        # If mask is provided, zero-out embeddings of padded points
        if mask is not None:
            mask = mask.unsqueeze(-1).float()   # (B, N, 1)
            point_emb = point_emb * mask        # mask out padded points' embeddings
            # Compute mean only over actual points (avoid counting pads)
            sum_emb = point_emb.sum(dim=1)      # (B, hidden_dim)
            count = mask.sum(dim=1)             # (B, 1) number of real points per sample
            global_emb = sum_emb / (count + 1e-8)
        else:
            # Simple average pooling over points
            global_emb = point_emb.mean(dim=1)  # (B, hidden_dim)
        return global_emb  # one embedding vector per sample


class NeutralizationModel(nn.Module):
    def __init__(self, backbone='se3', num_input_features=5, backbone_params=None):
        """
        backbone: 'se3' for SE(3) Transformer, 'mlp' for simple point MLP baseline.
        num_input_features: number of features per atom (not including coordinate dimensions).
                             E.g., 5 if using element one-hot only, 6 if including chain flag.
        backbone_params: dict of parameters for the backbone model.
           For SE(3) Transformer, could include 'dim', 'depth', etc.
           For MLP, could include 'hidden_dim', 'dropout'.
        """
        super(NeutralizationModel, self).__init__()
        if backbone_params is None:
            backbone_params = {}
        self.backbone_type = backbone
        if backbone == 'se3':
            if SE3Transformer is None:
                raise ImportError("SE3Transformer library is not installed.")
            # Set default parameters for SE3 Transformer if not provided
            model_dim = backbone_params.get('dim', 32)       # embedding dimension for features
            depth = backbone_params.get('depth', 4)          # number of attention layers
            num_heads = backbone_params.get('heads', 4)      # number of attention heads
            # We may need to embed input features to match model_dim if they differ
            self.feat_embed = nn.Identity()
            if num_input_features != model_dim:
                self.feat_embed = nn.Linear(num_input_features, model_dim)
            # Initialize SE3 Transformer model
            self.backbone = SE3Transformer(
                dim=model_dim,
                depth=depth,
                heads=num_heads,
                dim_head=backbone_params.get('dim_head', 64),
                num_degrees=backbone_params.get('num_degrees', 4),
                valid_radius=backbone_params.get('valid_radius', None)  # radius cutoff for attention, optional
            )
            self.embed_dim = model_dim  # output feature dimension per point
        elif backbone == 'mlp':
            hidden_dim = backbone_params.get('hidden_dim', 128)
            dropout = backbone_params.get('dropout', 0.1)
            # For the MLP, we'll include coordinates (3 dims) + input features per atom
            mlp_input_dim = 3 + num_input_features
            self.backbone = SimplePointNet(input_dim=mlp_input_dim, hidden_dim=hidden_dim, dropout=dropout)
            self.embed_dim = hidden_dim  # output dimension from the SimplePointNet
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}")
        # Classification head: a linear layer to predict the binary label from the global embedding
        self.classifier = nn.Linear(self.embed_dim, 1)

    def forward(self, coords, features, mask=None):
        # coords: (B, N, 3), features: (B, N, F), mask: (B, N) boolean
        if self.backbone_type == 'se3':
            # Embed features to match model dim if needed
            feat_emb = self.feat_embed(features)  # (B, N, model_dim) if a Linear is applied
            # Apply SE3 Transformer: output shape (B, N, embed_dim)
            point_features = self.backbone(feat_emb, coords, mask)
            # Mask out padded points (if any) and pool (mean) to get global features
            if mask is not None:
                mask_f = mask.unsqueeze(-1).float()          # (B, N, 1)
                point_features = point_features * mask_f     # zero out features for padded points
                sum_feats = point_features.sum(dim=1)        # (B, embed_dim)
                count = mask_f.sum(dim=1)                    # (B, 1)
                global_feat = sum_feats / (count + 1e-8)
            else:
                global_feat = point_features.mean(dim=1)     # (B, embed_dim)
        elif self.backbone_type == 'mlp':
            # The SimplePointNet backbone directly returns a global embedding per sample
            global_feat = self.backbone(coords, features, mask)  # (B, embed_dim)
        # Apply the classification head
        logit = self.classifier(global_feat).squeeze(-1)  # (B,) raw score for neutralization
        return logit
