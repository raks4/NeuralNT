import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import shutil
import cpuinfo
import pandas as pd
import subprocess


from model_builder import build_model
from data_loader import load_data
from validation import validate_model_forward_pass
from visualization import get_flat_weights, generate_3d_animation_pca
from layers import layer_configs, update_architecture_text
from torch.utils.data import DataLoader, TensorDataset

# This line sets the GPU as the device to train on otherwise sets the device to CPU if no GPU found
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.abspath("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)



# This function gets the device status shown on the frontend. 
# Input: Nothing
# Output: The device used and any potential cuda errors
def get_device_status():
    if not torch.cuda.is_available():
        return f"🔴 CPU: {cpuinfo.get_cpu_info()}"
    try:
        device_name = torch.cuda.get_device_name(0)
        mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)  
        mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)  
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  
        mem_free = total_mem - mem_reserved

        try:
            torch.cuda.synchronize()
            sync_status = "✅ GPU context healthy."
        except Exception as sync_error:
            sync_status = f"❌ GPU context error: {sync_error}"

        return (
            f"✅ CUDA Device: {device_name}<br>"
            f"Memory allocated: {mem_allocated:.2f} MB<br>"
            f"Memory reserved: {mem_reserved:.2f} MB<br>"
            f"Free memory (estimated): {mem_free:.2f} MB<br>"
            f"{sync_status}"
        )

    except Exception as e:
        return f"❌ CUDA error: {str(e)}"

# This is the function that trains the model. It calls many helper functions also and checks many parameters the user passes in from the frontend
# Input: loss_name, opt_name, lr, batch_size='32', image_size='28', file=None, custom_path=None, epochs='100', num_channels=3, generate_animation=False,  target_frames=None, frame_rate=None
# Output: Warnings, Info and inputs to graph functions and model for user to download. It will also delete the folder that was created to unzip files into at the end of training 
def train_model(loss_name, opt_name, lr, batch_size='32', image_size='28', file=None, custom_path=None, epochs='100', num_channels=3, generate_animation=False,  target_frames=None, frame_rate=None):

    animation_path = None
    loss_plot_path = None
    model_path = None
    final_logs = ""
    try: 
        target_frames = int(target_frames)
        if target_frames<=0:
            yield None, None, None, update_architecture_text(), "❌ Target frames must be a positive integer."
            return
    except:
        yield None, None, None, update_architecture_text(), "❌ Target frames must be numeric."
        return

    try:
        frame_rate = int(frame_rate)
        if frame_rate<=0:
            yield None, None, None, update_architecture_text(), "❌ Frame Rate must be a positive integer."
            return
    except:
        yield None, None, None, update_architecture_text(), "❌ Frame Rate must be numeric."
        return
            
            
    try:
        channels = int(num_channels)
        if channels not in [1, 3]:
            yield None, None, None, update_architecture_text(), "❌ Channels must be 1 or 3."
            return
    except:
        yield None, None, None, update_architecture_text(), "❌ Channels must be numeric."
        return

    # 1) Convert epochs to int with early validation
    try:
        max_epochs = int(epochs)
        if max_epochs <= 0:
            # Immediately yield error and stop
            yield None, None, None, update_architecture_text(), "❌ Epochs must be a positive integer."
            return
    except:
        yield None, None, None, update_architecture_text(), "❌ Epochs must be a valid number."
        return

    try:
        # 2) If no model is configured
        if not layer_configs:
            yield None, None, None, update_architecture_text(), "❌ No model configured! Please add at least one trainable layer."
            return

        # 3) If file path is invalid
        if not os.path.exists(file):
            msg = f"❌ File not found: {file}\nPlease check the path and try again."
            yield None, None, None, update_architecture_text(), msg
            return

        # 4) If file type is not .csv or .zip
        if not (file.endswith('.csv') or file.endswith('.zip')):
            yield None, None, None, update_architecture_text(), "❌ Invalid file type. Please provide a .csv or .zip file."
            return

        # 5) If custom directory is invalid
        if custom_path and not os.path.isdir(custom_path):
            try:
                os.makedirs(custom_path, exist_ok=True)
            except Exception as e:
                msg = f"❌ Could not create directory '{custom_path}': {e}"
                yield None, None, None, update_architecture_text(), msg
                return

        # 6) Build model
        lr = float(lr)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()  # Will raise if CUDA state is bad
            except RuntimeError as e:
                raise gr.Error("❌ CUDA failure detected. Please restart the dashboard or kernel.")

        model = build_model().to(device)

        # 7) Validate forward pass
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            if 'y' not in df.columns:
                yield None, None, None, update_architecture_text(), "❌ CSV missing 'y' column."
                return
            num_features = df.shape[1] - 1
            is_valid, error_msg, bad_layer_idx = validate_model_forward_pass(model, "tabular", num_features=num_features)
        else:
            is_valid, error_msg, bad_layer_idx = validate_model_forward_pass(model, "image", image_size=int(image_size), num_channels=num_channels)

        if not is_valid:
            # highlight offending layer + yield
            updated_view = update_architecture_text(highlight_index=bad_layer_idx)
            yield None, None, None, updated_view, ""
            return

        # 8) Check for trainable params
        if not any(p.requires_grad for p in model.parameters()):
            yield None, None, None, update_architecture_text(), "⚠️ Model has no trainable parameters. Add a Linear or Conv2d layer."
            return

        # 9) Validate batch_size
        try:
            batch_size = int(batch_size)
            if batch_size <= 0:
                yield None, None, None, update_architecture_text(), "❌ Batch size must be a positive integer"
                return
        except:
            yield None, None, None, update_architecture_text(), "❌ Batch size must be a valid number"
            return

        # 10) Validate image_size
        try:
            image_size = int(image_size)
            if image_size <= 0:
                yield None, None, None, update_architecture_text(), "❌ Image size must be a positive integer"
                return
        except:
            yield None, None, None, update_architecture_text(), "❌ Image size must be a valid number"
            return

        # 11) Load data
        loss_fn = nn.MSELoss() if loss_name == 'MSELoss' else nn.CrossEntropyLoss()
        data = load_data(file, custom_path, batch_size=batch_size, image_size=image_size, num_channels=channels, loss_fn = loss_fn)
        optimizer = optim.SGD(model.parameters(), lr=lr) if opt_name == 'SGD' else optim.Adam(model.parameters(), lr=lr)

        loss_history = []
        weight_path = []
        status_lines = []
        
        # 12) Train

        if data["type"] == "tabular":
            X_train, y_train = data["train"]
        
            # Convert full dataset tensors (before DataLoader)
            X_train = torch.tensor(X_train, dtype=torch.float32)
        
            if isinstance(loss_fn, nn.MSELoss):
                y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                y_train = torch.tensor(y_train, dtype=torch.long).view(-1)  
            elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  
            else:
                yield None, None, None, update_architecture_text(), "❌ Error loading training data"
                return
        
            # DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
            for epoch in range(1, max_epochs + 1):
                epoch_loss = 0
                num_batches = 0
        
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
        
                    optimizer.zero_grad()
        
                    if isinstance(loss_fn, nn.MSELoss):
                        y_input = y_batch.float().view(-1, 1)
                    elif isinstance(loss_fn, nn.CrossEntropyLoss):
                        y_input = y_batch.long().view(-1)  # CrossEntropy expects class indices, not 2D
                    elif isinstance(loss_fn, nn.BCEWithLogitsLoss):
                        y_input = y_batch.float().view(-1, 1)
                    else:
                        yield None, None, None, update_architecture_text(), "❌ Unhandled Loss Function"
                        return
        
                    
                    out = model(X_batch)
                    loss = loss_fn(out, y_input)
                    loss.backward()
                    optimizer.step()
        
                    epoch_loss += loss.item()
                    num_batches += 1
        
                    weight_path.append(get_flat_weights(model).cpu().numpy())
        
                avg_epoch_loss = epoch_loss / num_batches
                loss_history.append(avg_epoch_loss)
        
                status_lines.append(f"Epoch {epoch}/{max_epochs} — Loss: {avg_epoch_loss:.4f}")
                yield None, None, None, update_architecture_text(), "\n\n".join(status_lines)

        else:
            # image data
            train_loader = data["train"]
            for epoch in range(1, max_epochs + 1):
                epoch_loss = 0
                num_batches = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    if isinstance(loss_fn, nn.MSELoss):
                        # One-hot for MSE
                        y_input = torch.nn.functional.one_hot(
                            y_batch,
                            num_classes=len(torch.unique(y_batch))
                        ).float().to(device)
                    else:
                        y_input = y_batch

                    out = model(X_batch)
                    if isinstance(loss_fn, nn.MSELoss):
                        out = torch.softmax(out, dim=1)

                    loss = loss_fn(out, y_input)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    weight_path.append(get_flat_weights(model).cpu().numpy())


                # Average loss for the epoch
                avg_epoch_loss = epoch_loss / num_batches
                loss_history.append(avg_epoch_loss)

                status_lines.append(f"Epoch {epoch}/{max_epochs} — Loss: {avg_epoch_loss:.4f}")
                yield None, None, None, update_architecture_text(), "\n\n".join(status_lines)

            # Cleanup extracted images
            if data["path"]:
                try:
                    shutil.rmtree(data["path"])
                except Exception as e:
                    gr.Warning(f"Warning: Could not delete extracted folder: {e}")

        loss_plot_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
        fig, ax = plt.subplots()
        ax.plot(loss_history)
        ax.set_title("Loss over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.savefig(loss_plot_path)
        plt.close(fig)
        animation_path = os.path.join(OUTPUT_DIR, "animation.mp4")


        if generate_animation:

            generate_3d_animation_pca(
            np.array(weight_path),
            loss_history,
            animation_path, 
            target_frames=target_frames,
            frame_rate=frame_rate
            )
        else:
            create_dummy_video(animation_path)
        # Save the trained model in OUTPUT_DIR
        model_path = os.path.join(OUTPUT_DIR, "trained_model.pt")
        torch.save(model, model_path)

        final_logs = "\n".join(status_lines)

        for path in [loss_plot_path, animation_path, model_path]:
            if os.path.isdir(path):
                gr.Warning(f"❌ Output path is a directory, expected a file: {path}")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        yield loss_plot_path,  animation_path, model_path, update_architecture_text(), final_logs

    except gr.Error as e:
        raise e
    except Exception as e:
        raise gr.Error(f"❌ Unexpected error: {str(e)}")


# This function wraps the training operation, ensures a valid save path, and yields training outputs
# Input: loss_name, opt_name, lr, batch_size, image_size, 
                                  #file, custom_path, epochs, num_channels, 
                                 # generate_animation, target_frames, frame_rate
# Output: A generator which returns training progress outputs from train_model
def train_model_with_default_path(loss_name, opt_name, lr, batch_size, image_size, 
                                  file, custom_path, epochs, num_channels, 
                                  generate_animation, target_frames, frame_rate
):
    if not custom_path or custom_path.strip() == "":
        custom_path = get_default_writable_folder()

    for item in train_model(loss_name, opt_name, lr, 
                            batch_size, image_size, file, 
                            custom_path, epochs, num_channels, 
                            generate_animation, target_frames, frame_rate
    ):
        yield item




# This function gets a subfolder in the home directory which is not protected and is writeable
# Input: Nothing
# Output: A writeable folder
def get_default_writable_folder():

    home_dir = os.path.expanduser("~")  # e.g. C:\\Users\\Alice on Windows
    default_path = os.path.join(home_dir, "my_gradio_data")
    os.makedirs(default_path, exist_ok=True)
    return default_path


# This function creates a dummy video to prevent the gradio frontend from crashing if the user selects no video
# Input: Output_path to put the video in
# Output: A black video for the gradio screen
def create_dummy_video(output_path):
    command = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:d=5", 
        "-c:v", "libx264",
        "-t", "5",
        "-pix_fmt", "yuv420p",
        "-y", output_path
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        gr.Warning(f"❌ Error creating dummy video: {e.stderr.decode()}")
