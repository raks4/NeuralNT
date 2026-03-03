import torch
import torch.nn as nn
import gradio as gr

# This line sets the GPU as the device to train on otherwise sets the device to CPU if no GPU found
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# --------------------------------
# Training Functions
# --------------------------------

# This function makes sure that the model being built is a valid model before training
# Input: model, data_type, image_size=28, num_features=None, num_channels=3
# Output: errors if the model is not valid and a three tuple if the model is valid
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

# This function validates the entire pipeline through a series of tests
# Input: X_tensor, y_tensor, model, loss_fn, batch_size=None, auto_fix=True
# Output: A gr.Warning if any of the tests is failed and X_tensor, y_tensor which are potentially modified tensors after validation and auto-fixes
def full_pipeline_validator(X_tensor, y_tensor, model, loss_fn, batch_size=None, auto_fix=True):

    if X_tensor.shape[0] == 0 or y_tensor.shape[0] == 0:
        gr.Warning("❌ Dataset is empty. Check your data loading pipeline.")


    if torch.isnan(X_tensor).any() or torch.isnan(y_tensor).any():
        gr.Warning("❌ NaN detected in input or target tensor.")
    if torch.isinf(X_tensor).any() or torch.isinf(y_tensor).any():
        gr.Warning("❌ Inf detected in input or target tensor.")
    
    gr.Info("✅ No NaNs or Infs detected.")


    if X_tensor.dtype != torch.float32:
        if auto_fix:
            gr.Info(f"⚠️ Auto-fixing X_tensor dtype from {X_tensor.dtype} to torch.float32")
            X_tensor = X_tensor.float()
        else:
            gr.Warning(f"❌ X_tensor dtype must be float32, got {X_tensor.dtype}")

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        if y_tensor.dtype != torch.long:
            if auto_fix:
                gr.Info(f"⚠️ Auto-fixing y_tensor dtype from {y_tensor.dtype} to torch.long")
                y_tensor = y_tensor.long()
            else:
                gr.Warning(f"❌ For CrossEntropyLoss, y_tensor dtype must be torch.long, got {y_tensor.dtype}")
    else:
        if y_tensor.dtype != torch.float32:
            if auto_fix:
                gr.Info(f"⚠️ Auto-fixing y_tensor dtype from {y_tensor.dtype} to torch.float32")
                y_tensor = y_tensor.float()
            else:
                gr.Warning(f"❌ y_tensor dtype must be float32, got {y_tensor.dtype}")

    if len(X_tensor.shape) != 2:
        gr.Warning(f"❌ X_tensor must be 2D [batch_size, features], got {X_tensor.shape}")

    if isinstance(loss_fn, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        expected_y_shape = (X_tensor.shape[0], 1)
        if y_tensor.shape != expected_y_shape:
            if auto_fix:
                gr.Info(f"⚠️ Auto-reshaping y_tensor from {y_tensor.shape} to {expected_y_shape}")
                y_tensor = y_tensor.view(-1, 1)
            else:
                gr.Warning(f"❌ y_tensor should have shape {expected_y_shape}, but got {y_tensor.shape}")

    elif isinstance(loss_fn, nn.CrossEntropyLoss):
        expected_y_shape = (X_tensor.shape[0],)
        if y_tensor.shape != expected_y_shape:
            if auto_fix:
                gr.Info(f"⚠️ Auto-reshaping y_tensor from {y_tensor.shape} to {expected_y_shape}")
                y_tensor = y_tensor.view(-1)
            else:
                gr.Warning(f"❌ y_tensor should have shape {expected_y_shape}, but got {y_tensor.shape}")

    if batch_size is not None and X_tensor.shape[0] < batch_size:
        gr.Info(f"⚠️ Batch size {batch_size} is larger than dataset size {X_tensor.shape[0]}, adjusting.")
        batch_size = X_tensor.shape[0]

    try:
        model.eval()  
        with torch.no_grad():
            dummy_out = model(X_tensor[:1])
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                if dummy_out.ndim != 2:
                    gr.Warning(f"❌ Model output for CrossEntropyLoss should be 2D [batch_size, num_classes], got {dummy_out.shape}")
            else:
                if dummy_out.shape != y_tensor[:1].shape:
                    gr.Warning(f"❌ Model output shape {dummy_out.shape} does not match target shape {y_tensor[:1].shape}")
    except Exception as e:
        gr.Warning(f"❌ Model forward pass failed: {e}")

 
    return X_tensor, y_tensor
