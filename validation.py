import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model_forward_pass(model, data_type, image_size=28, num_features=None, num_channels=3):
    try:
        if data_type == "tabular":
            assert num_features is not None, "Missing number of input features for tabular data"
            dummy_input = torch.randn(1, num_features).to(device)
        else:
            dummy_input = torch.randn(1, num_channels, image_size, image_size).to(device)

        x = dummy_input
        for idx, layer in enumerate(model):
            try:
                x = layer(x)
            except Exception as e:
                return False, f"Shape mismatch at layer {idx}: {layer.__class__.__name__} — {str(e)}", idx
        return True, None, None
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}", None

def full_pipeline_validator(X_tensor, y_tensor, model, loss_fn, batch_size=None, auto_fix=True):
    if X_tensor.shape[0] == 0 or y_tensor.shape[0] == 0:
        print("Warning: Dataset is empty.")

    if torch.isnan(X_tensor).any() or torch.isnan(y_tensor).any():
        print("Warning: NaN detected in input or target tensor.")
    
    if X_tensor.dtype != torch.float32:
        if auto_fix:
            X_tensor = X_tensor.float()

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        if y_tensor.dtype != torch.long:
            if auto_fix:
                y_tensor = y_tensor.long()
    else:
        if y_tensor.dtype != torch.float32:
            if auto_fix:
                y_tensor = y_tensor.float()

    return X_tensor, y_tensor
