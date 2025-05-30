import torch
import torch.nn as nn
from models.mamba_net import JetImageMamba

def get_model(data_config, **kwargs):
    """
    Required function for Weaver model configuration
    
    Args:
        data_config: Weaver data configuration object
        **kwargs: Additional arguments
    
    Returns:
        model: PyTorch model instance
        model_info: Dictionary with model metadata
    """
    
    # Get model parameters from kwargs (passed from command line or config)
    model_params = kwargs.get('model_params', {})
    
    # Create model instance
    model = JetImageMamba(
        in_channels=model_params.get('in_channels', 12),
        num_classes=model_params.get('num_classes', 10),
        d_model=model_params.get('d_model', 128),
        n_layers=model_params.get('n_layers', 6),
        d_state=model_params.get('d_state', 16),
        dropout=model_params.get('dropout', 0.1)
    )
    
    # Model info for ONNX export and other purposes
    model_info = {
        'input_names': ['jet_images'],
        'input_shapes': {
            'jet_images': (None, model_params.get('in_channels', 12), 33, 33)  # (batch, channels, height, width)
        },
        'output_names': ['logits'],
        'dynamic_axes': {
            'jet_images': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    }
    
    return model, model_info

def get_loss(data_config, **kwargs):
    """
    Optional function for custom loss function
    If not provided, Weaver uses torch.nn.CrossEntropyLoss()
    """
    label_smoothing = kwargs.get('label_smoothing', 0.0)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
