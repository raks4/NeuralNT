import torch.nn as nn
from utils import parse_int_or_tuple

# The dictionary containing a map of all layers.
layer_map = {
    "Linear": lambda in_dim, out_dim: nn.Linear(int(in_dim), int(out_dim)),
    "Conv2d": lambda in_dim, out_dim: nn.Conv2d(int(in_dim), int(out_dim), kernel_size=3, padding=1),  
    "MaxPool2d": lambda *_: nn.MaxPool2d(kernel_size=2),  
    "AvgPool2d": lambda *_: nn.AvgPool2d(kernel_size=2),  
    "Dropout": lambda p=0.5, *_: nn.Dropout(float(p)),
    "ReLU": lambda *_: nn.ReLU(),
    "Tanh": lambda *_: nn.Tanh(),
    "Sigmoid": lambda *_: nn.Sigmoid(),
    "Flatten": lambda *_: nn.Flatten(),
    "Softmax": lambda *_: nn.Softmax(dim=1),
    "LeakyReLU": lambda slope=0.01, *_: nn.LeakyReLU(negative_slope=float(slope)),
    "GELU": lambda *_: nn.GELU(),
    "ELU": lambda alpha=1.0, *_: nn.ELU(alpha=float(alpha))
}

layer_configs = []

def validate_layer_inputs(layer_type, **kwargs):
    try:
        if layer_type == "Linear":
            in_dim_int = int(kwargs.get("in_dim") or 0)
            out_dim_int = int(kwargs.get("out_dim") or 0)
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} dimensions must be positive integers"
        elif layer_type == "Conv2d":
            in_dim_int = int(kwargs.get("in_dim") or 0)
            out_dim_int = int(kwargs.get("out_dim") or 0)
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} in/out dims must be positive integers"
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def add_layer(
    layer_type, in_dim, out_dim,
    kernel_size=3, padding=1, stride=1, bias=1,
    pool_kernel="2", pool_stride="2", pool_padding="0",
    avgpool_kernel=None, avgpool_stride=None, avgpool_padding=None,
    leaky_slope = "0.01", elu_alpha = "1.0"
):
    is_valid, err_msg = validate_layer_inputs(layer_type=layer_type, in_dim=in_dim, out_dim=out_dim)
    if not is_valid: return err_msg
    
    desc = f"{layer_type}({in_dim}, {out_dim})"
    config = (desc, layer_type, in_dim, out_dim, kernel_size, padding, stride, bias)
    layer_configs.append(config)
    return update_architecture_text()

def update_layer(index, layer_type, in_dim, out_dim, *args, **kwargs):
    index = int(index)
    if 0 <= index < len(layer_configs):
        desc = f"{layer_type}({in_dim}, {out_dim})"
        layer_configs[index] = (desc, layer_type, in_dim, out_dim, None, None, None, None)
    return update_architecture_text()

def delete_layer(index):
    index = int(index)
    if 0 <= index < len(layer_configs):
        layer_configs.pop(index)
    return update_architecture_text()

def reset_layers():
    layer_configs.clear()

def update_architecture_text(highlight_index=None):
    lines = []
    for i, config in enumerate(layer_configs):
        prefix = f"{i}: "
        desc = config[0]
        if i == highlight_index:
            desc = f"⚠️ {desc}"
        lines.append(prefix + desc)
    return "\n".join(lines)
