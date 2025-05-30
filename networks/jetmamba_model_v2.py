"""
JetMamba Network Configuration for JetClass Data
Uses the standard JetClass_full.yaml input format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMambaClassifier(nn.Module):
    """Simplified Mamba-inspired classifier for JetClass data"""
    
    def __init__(self, input_dims, num_classes, hidden_dim=128):
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dims, hidden_dim)
        
        # Simple "Mamba-like" layers (using GRU as placeholder for now)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, *args, **kwargs):
        # Handle both positional and keyword arguments for compatibility with FLOPS counter
        if args:
            # When called with positional arguments (from FLOPS counter)
            # args should be in the order specified by model_info['input_names']
            # which is ['pf_points', 'pf_features', 'pf_vectors', 'pf_mask']
            if len(args) >= 4:
                pf_features = args[1]  # pf_features is the second input
                pf_mask = args[3]      # pf_mask is the fourth input
            else:
                # Fallback if not enough arguments
                raise ValueError(f"Expected at least 4 positional arguments, got {len(args)}")
        else:
            # When called with keyword arguments (normal case)
            pf_features = kwargs['pf_features']  # (B, C=17, N=128)
            pf_mask = kwargs['pf_mask']         # (B, 1, N=128)
        
        # Transpose pf_features from (B, C, N) to (B, N, C) format
        pf_features = pf_features.transpose(1, 2)  # (B, N=128, C=17)
        
        # Transpose and squeeze mask: (B, 1, N) -> (B, N, 1) -> (B, N)
        pf_mask = pf_mask.transpose(1, 2)  # (B, N=128, 1)
        mask = pf_mask.squeeze(-1)  # (B, N=128)
        
        print(f"Input shapes - pf_features: {pf_features.shape}, pf_mask: {pf_mask.shape}, mask: {mask.shape}")
        
        # Embed inputs
        x = self.embedding(pf_features)  # (B, N=128, hidden_dim)
        
        # Apply GRU layers (Mamba placeholder)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        
        # Attention pooling
        attention_weights = self.attention(x)  # (B, N, 1)
        attention_weights = attention_weights * mask.unsqueeze(-1)  # Apply mask
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted average
        x_pooled = (x * attention_weights).sum(dim=1)  # (B, hidden_dim*2)
        
        # Classification
        logits = self.classifier(x_pooled)
        
        return logits


def get_model(data_config, **kwargs):
    """
    Model factory function for Weaver with JetClass data format
    """
    
    print("="*50)
    print("DEBUG: JetClass Data config information")
    print(f"Available input names: {list(data_config.input_names)}")
    print(f"Available input shapes: {data_config.input_shapes}")
    print(f"Label names: {data_config.label_names}")
    print("="*50)
    
    # Use pf_features as main input (this contains the 17 particle features)
    if 'pf_features' not in data_config.input_shapes:
        raise ValueError(f"Expected 'pf_features' in input shapes, got: {list(data_config.input_shapes.keys())}")
    
    input_dims = data_config.input_shapes['pf_features'][1]  # Should be 17
    #num_classes = len(data_config.label_names)
    num_classes=len(data_config.label_value)
    hidden_dim = kwargs.get('hidden_dim', 128)
    
    print(f"Creating SimpleMambaClassifier for JetClass:")
    print(f"  - Input dims (pf_features): {input_dims}")
    print(f"  - Num classes: {num_classes}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Input shape pf_features: {data_config.input_shapes['pf_features']}")
    print(f"  - Input shape pf_mask: {data_config.input_shapes.get('pf_mask', 'Not found')}")
    
    model = SimpleMambaClassifier(input_dims, num_classes, hidden_dim)
    
    # Model info for Weaver
    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: f'n_{k}'} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }
    
    return model, model_info


def get_loss(data_config, **kwargs):
    """Loss function for classification"""
    return torch.nn.CrossEntropyLoss()