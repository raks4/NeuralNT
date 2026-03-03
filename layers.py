import gradio as gr
import torch.nn as nn
from utils import parse_int_or_tuple

# The dictionary containing a map of all layers. Used in wiring the frontend
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

# This function checks all dimensions and makes sures correct dimensions are being passed in. 
# Input: A layer type and dimensions passed in.
# Output: Nothing if correct and false otherwise along with a message
def validate_layer_inputs(layer_type, **kwargs):
    try:
        if layer_type == "Linear":
            in_dim_int = int(kwargs.get("in_dim"))
            out_dim_int = int(kwargs.get("out_dim"))
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} dimensions must be positive integers"

        elif layer_type == "Conv2d":
            in_dim_int = int(kwargs.get("in_dim"))
            out_dim_int = int(kwargs.get("out_dim"))
            kernel_dim = parse_int_or_tuple(kwargs.get("kernel_size", 3))
            padding_dim = parse_int_or_tuple(kwargs.get("padding", 1))
            stride = parse_int_or_tuple(kwargs.get("stride", 1))

            for val in [kernel_dim, padding_dim, stride]:
                if isinstance(val, tuple):
                    if any(v < 0 for v in val):
                        return False, f"{layer_type} tuple values must be non-negative"
                else:
                    if val < 0:
                        return False, f"{layer_type} values must be non-negative"

            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} in/out dims must be positive integers"

        elif layer_type == "Dropout":
            p = float(kwargs.get("in_dim"))
            if not (0 <= p <= 1):
                return False, "Dropout probability must be between 0 and 1"

        elif layer_type == "MaxPool2d":
            kernel = parse_int_or_tuple(kwargs.get("pool_kernel", 2))
            stride = parse_int_or_tuple(kwargs.get("pool_stride", 2))
            padding = parse_int_or_tuple(kwargs.get("pool_padding", 0))
            for val in [kernel, stride, padding]:
                if isinstance(val, tuple):
                    if any(v < 0 for v in val):
                        return False, f"{layer_type} tuple values must be non-negative"
                else:
                    if val < 0:
                        return False, f"{layer_type} values must be non-negative"
        
        elif layer_type == "AvgPool2d":
            kernel = parse_int_or_tuple(kwargs.get("avgpool_kernel", 2))
            stride = parse_int_or_tuple(kwargs.get("avgpool_stride", 2))
            padding = parse_int_or_tuple(kwargs.get("avgpool_padding", 0))
            for val in [kernel, stride, padding]:
                if isinstance(val, tuple):
                    if any(v < 0 for v in val):
                        return False, f"{layer_type} tuple values must be non-negative"
                else:
                    if val < 0:
                        return False, f"{layer_type} values must be non-negative"

        elif layer_type == "LeakyReLU":
            slope = float(kwargs.get("leaky_slope", 0.01))
            if slope < 0:
                return False, "LeakyReLU negative_slope must be ≥ 0"
        
        elif layer_type == "ELU":
            alpha = float(kwargs.get("elu_alpha", 1.0))
            if alpha < 0:
                return False, "ELU alpha must be ≥ 0"

        return True, None
    except Exception as e:
        return False, f"Validation error in {layer_type}: {str(e)}"


# This function adds layers as the user requests on the frontend
# Input: Parameters and type of layer passed in from the frontend
# Output: A layer of the type requested is added or an error message is passed to the frontend
def add_layer(
    layer_type, in_dim, out_dim,
    kernel_size=3, padding=1, stride=1, bias=1,
    pool_kernel="2", pool_stride="2", pool_padding="0",
    avgpool_kernel=None, avgpool_stride=None, avgpool_padding=None,
    leaky_slope = "0.01", elu_alpha = "1.0"
):

    is_valid, err_msg = validate_layer_inputs(
        layer_type=layer_type,
        in_dim=in_dim,
        out_dim=out_dim,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride,
        pool_padding=pool_padding,
        avgpool_kernel=avgpool_kernel, 
        avgpool_stride=avgpool_stride,
        avgpool_padding=avgpool_padding,
        leaky_slope=leaky_slope,
        elu_alpha=elu_alpha
    )

    if not is_valid:
        return err_msg

    try:
        if layer_type == "Conv2d":
            in_dim = int(in_dim)
            out_dim = int(out_dim)
            k = parse_int_or_tuple(kernel_size or 3)
            p = parse_int_or_tuple(padding or 1)
            s = parse_int_or_tuple(stride or 1)
            b = bool(bias)
            desc = f"Conv2d({in_dim}, {out_dim}, kernel={k}, padding={p}, stride={s}, bias={b})"
            config = (desc, layer_type, in_dim, out_dim, k, p, s, b)
        
        elif layer_type == "LeakyReLU":
            negative_slope = float(leaky_slope or "0.01")
            desc = f"LeakyReLU(negative_slope={negative_slope})"
            config = (desc, layer_type, negative_slope, negative_slope, None, None, None, None)
        
        elif layer_type == "ELU":
            alpha = float(elu_alpha or "1.0")
            desc = f"ELU(alpha={alpha})"
            config = (desc, layer_type, alpha, alpha, None, None, None, None)

        elif layer_type == "Softmax":
            desc = "Softmax(dim=1)"
            config = (desc, layer_type, None, None, None, None, None, None)

        elif layer_type == "GELU":
            desc = "GELU()"
            config = (desc, layer_type, None, None, None, None, None, None)
            
        elif layer_type == "Linear":
            in_dim = int(in_dim)
            out_dim = int(out_dim)
            desc = f"Linear({in_dim}, {out_dim})"
            config = (desc, layer_type, in_dim, out_dim, None, None, None, None)

        elif layer_type == "Dropout":
            p = float(in_dim)
            desc = f"Dropout({p})"
            config = (desc, layer_type, p, p, None, None, None, None)

        elif layer_type == "MaxPool2d":
            kernel_val = parse_int_or_tuple(pool_kernel or "2")
            stride_val = parse_int_or_tuple(pool_stride or "2")
            padding_val = parse_int_or_tuple(pool_padding or "0")
            desc = f"MaxPool2d(kernel={kernel_val}, stride={stride_val}, padding={padding_val})"
            config = (desc, layer_type, None, None, kernel_val, padding_val, stride_val, None)

        elif layer_type == "AvgPool2d":
            kernel_val = parse_int_or_tuple(avgpool_kernel or "2")
            stride_val = parse_int_or_tuple(avgpool_stride or "2")
            padding_val = parse_int_or_tuple(avgpool_padding or "0")
            desc = f"AvgPool2d(kernel={kernel_val}, stride={stride_val}, padding={padding_val})"
            config = (desc, layer_type, None, None, kernel_val, padding_val, stride_val, None)

        else:
            # For e.g. ReLU/Tanh/Sigmoid/Flatten
            desc = layer_type
            config = (desc, layer_type, None, None, None, None, None, None)
            
    except Exception as e:
        desc = f"[Error Adding Layer: {e}]"
        config = (desc, layer_type, None, None, None, None, None, None)

    layer_configs.append(config)
    return update_architecture_text()



# This function updates layers as the user requests on the frontend
# Input: Parameters and type of layer passed in from the frontend
# Output: A layer of the type requested is added or an error message is passed to the frontend
def update_layer(
    index, layer_type, in_dim, out_dim,
    kernel_size=3, padding=1, stride=1, bias=True,
    pool_kernel="2", pool_stride="2", pool_padding="0",
    avgpool_kernel=None, avgpool_stride=None, avgpool_padding=None,
    leaky_slope= "0.01",elu_alpha = "1.0"
):
    
    index = int(index)
    is_valid, err_msg = validate_layer_inputs(
        layer_type=layer_type,
        in_dim=in_dim,
        out_dim=out_dim,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride,
        pool_padding=pool_padding,
        avgpool_kernel=avgpool_kernel, 
        avgpool_stride=avgpool_stride,
        avgpool_padding=avgpool_padding,
        leaky_slope=leaky_slope,
        elu_alpha=elu_alpha
    )

    if not is_valid:
        return err_msg
    if index < 0 or index >= len(layer_configs):
        return update_architecture_text()

    try:
        if layer_type == "Conv2d":
            i = int(in_dim)
            o = int(out_dim)
            k = parse_int_or_tuple(kernel_size or 3)
            p = parse_int_or_tuple(padding or 1)
            s = parse_int_or_tuple(stride or 1)
            b = bool(bias)
            desc = f"Conv2d({i}, {o}, kernel={k}, padding={p}, stride={s}, bias={b})"
            layer_configs[index] = (desc, layer_type, i, o, k, p, s, b)

        elif layer_type == "Linear":
            i = int(in_dim)
            o = int(out_dim)
            desc = f"Linear({i}, {o})"
            layer_configs[index] = (desc, layer_type, i, o, None, None, None, None)
        
        elif layer_type == "ELU":
            alpha = float(elu_alpha or "1.0")
            desc = f"ELU(alpha={alpha})"
            layer_configs[index] = (desc, layer_type, alpha, alpha, None, None, None, None)

        elif layer_type == "GELU":
            desc = "GELU()"
            layer_configs[index] = (desc, layer_type, None, None, None, None, None, None)

        elif layer_type == "LeakyReLU":
            negative_slope = float(leaky_slope or "0.01")
            desc = f"LeakyReLU(negative_slope={negative_slope})"
            layer_configs[index] = (desc, layer_type, negative_slope, negative_slope, None, None, None, None)

        elif layer_type == "Softmax":
            desc = "Softmax(dim=1)"
            layer_configs[index] = (desc, layer_type, None, None, None, None, None, None)

        elif layer_type == "Dropout":
            p = float(in_dim)
            desc = f"Dropout({p})"
            layer_configs[index] = (desc, layer_type, p, p, None, None, None, None)

        elif layer_type == "AvgPool2d":
            kv = parse_int_or_tuple(avgpool_kernel or "2")
            sv = parse_int_or_tuple(avgpool_stride or "2")
            pv = parse_int_or_tuple(avgpool_padding or "0")
            desc = f"AvgPool2d(kernel={kv}, stride={sv}, padding={pv})"
            layer_configs[index] = (desc, layer_type, None, None, kv, pv, sv, None)

        elif layer_type == "MaxPool2d":
            kv = parse_int_or_tuple(pool_kernel or "2")
            sv = parse_int_or_tuple(pool_stride or "2")
            pv = parse_int_or_tuple(pool_padding or "0")
            desc = f"MaxPool2d(kernel={kv}, stride={sv}, padding={pv})"
            layer_configs[index] = (desc, layer_type, None, None, kv, pv, sv, None)

        else:
            desc = layer_type
            layer_configs[index] = (desc, layer_type, None, None, None, None, None, None)

    except Exception as e:
        layer_configs[index] = (f"[Error Editing Layer: {e}]", layer_type, None, None, None, None, None, None)

    return update_architecture_text()


# This function inserts layers as the user requests on the frontend
# Input: Parameters and type of layer passed in from the frontend
# Output: A layer of the type requested is added or an error message is passed to the frontend
def insert_layer(
    index, layer_type, in_dim, out_dim,
    kernel_size=3, padding=1, stride=1, bias=1,
    pool_kernel="2", pool_stride="2", pool_padding="0",
    avgpool_kernel=None, avgpool_stride=None, avgpool_padding=None, 
    leaky_slope="0.01", elu_alpha = "1.0"
):
    index = int(index)
    is_valid, err_msg = validate_layer_inputs(
        layer_type=layer_type,
        in_dim=in_dim,
        out_dim=out_dim,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride,
        pool_padding=pool_padding,
        avgpool_kernel=avgpool_kernel, 
        avgpool_stride=avgpool_stride,
        avgpool_padding=avgpool_padding,
        leaky_slope=leaky_slope,
        elu_alpha=elu_alpha
    )

    if not is_valid:
        return err_msg

    try:
        if layer_type == "Conv2d":
            i = int(in_dim)
            o = int(out_dim)
            k = parse_int_or_tuple(kernel_size or "3")
            p = parse_int_or_tuple(padding or "1")
            s = parse_int_or_tuple(stride or "1")
            b = bool(bias)
            desc = f"Conv2d({i}, {o}, kernel={k}, padding={p}, stride={s}, bias={b})"
            layer_configs.insert(index, (desc, layer_type, i, o, k, p, s, b))

        elif layer_type == "Linear":
            i = int(in_dim)
            o = int(out_dim)
            desc = f"Linear({i}, {o})"
            layer_configs.insert(index, (desc, layer_type, i, o, None, None, None, None))

        elif layer_type == "ELU":
            alpha = float(elu_alpha or "1.0")
            desc = f"ELU(alpha={alpha})"
            layer_configs.insert(index, (desc, layer_type, alpha, alpha, None, None, None, None))

        elif layer_type == "LeakyReLU":
            negative_slope = float(leaky_slope or "0.01")
            desc = f"LeakyReLU(negative_slope={negative_slope})"
            layer_configs.insert(index, (desc, layer_type, negative_slope, negative_slope, None, None, None, None))

        elif layer_type == "Softmax":
            desc = "Softmax(dim=1)"
            layer_configs.insert(index, (desc, layer_type, None, None, None, None, None, None))

        elif layer_type == "Dropout":
            p = float(in_dim)
            desc = f"Dropout({p})"
            layer_configs.insert(index, (desc, layer_type, p, p, None, None, None, None))

        elif layer_type == "MaxPool2d":
            kv = parse_int_or_tuple(pool_kernel or "2")
            sv = parse_int_or_tuple(pool_stride or "2")
            pv = parse_int_or_tuple(pool_padding or "0")
            desc = f"MaxPool2d(kernel={kv}, stride={sv}, padding={pv})"
            layer_configs.insert(index, (desc, layer_type, None, None, kv, pv, sv, None))

        elif layer_type == "AvgPool2d":
            kv = parse_int_or_tuple(avgpool_kernel or "2")
            sv = parse_int_or_tuple(avgpool_stride or "2")
            pv = parse_int_or_tuple(avgpool_padding or "0")
            desc = f"AvgPool2d(kernel={kv}, stride={sv}, padding={pv})"
            layer_configs.insert(index, (desc, layer_type, None, None, kv, pv, sv, None))

        elif layer_type == "GELU":
            desc = "GELU()"
            layer_configs.insert(index, (desc, layer_type, None, None, None, None, None, None))

        else:
            desc = layer_type
            layer_configs.insert(index, (desc, layer_type, None, None, None, None, None, None))

    except Exception as e:
        desc = f"[Error Inserting Layer: {e}]"
        layer_configs.insert(index, (desc, layer_type, None, None, None, None, None, None))

    return update_architecture_text()



# This function deletes layers as the user requests on the frontend in the edit tab
def delete_layer(index):
    index = int(index)
    if 0 <= index < len(layer_configs):
        layer_configs.pop(index)
    return update_architecture_text()
    
# This function clears all layers currently selected if the user hits reset on the frontend
def reset_layers():
    layer_configs.clear()
    return ""

# This helper function shows the error messages if dimensions do not match. It does not calculate them it is simply called to show the visual error message. 
# Input: index to highlight at
# Output: A warning printed on the architecture at the mismatch layer
def update_architecture_text(highlight_index=None):
    lines = []
    for i, config in enumerate(layer_configs):
        prefix = f"{i}: "
        desc = config[0]
        if i == highlight_index:
            desc = f"⚠️ {desc}"
        lines.append(prefix + desc)
    return "\n".join(lines)
