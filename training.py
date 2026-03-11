import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.abspath("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_device_status():
    if not torch.cuda.is_available():
        return f"CPU: {cpuinfo.get_cpu_info()}"
    return f"CUDA Device: {torch.cuda.get_device_name(0)}"

def train_model(loss_name, opt_name, lr, batch_size='32', image_size='28', file=None, custom_path=None, epochs='100', num_channels=3, generate_animation=False, target_frames=300, frame_rate=10):
    try:
        max_epochs = int(epochs)
        batch_size = int(batch_size)
        image_size = int(image_size)
        lr = float(lr)
        channels = int(num_channels)

        model = build_model().to(device)

        # Load data
        loss_fn = nn.MSELoss() if loss_name == 'MSELoss' else nn.CrossEntropyLoss()
        data = load_data(file, custom_path, batch_size=batch_size, image_size=image_size, num_channels=channels, loss_fn=loss_fn)
        optimizer = optim.SGD(model.parameters(), lr=lr) if opt_name == 'SGD' else optim.Adam(model.parameters(), lr=lr)

        loss_history = []
        status_lines = []

        train_loader = data["train"]
        if data["type"] == "tabular":
            X_train, y_train = train_loader
            train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, max_epochs + 1):
            epoch_loss = 0
            num_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                out = model(X_batch)
                loss = loss_fn(out, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            log = f"Epoch {epoch}/{max_epochs} — Loss: {avg_loss:.4f}"
            print(log)
            yield None, None, None, None, log

        # Save results
        loss_plot_path = os.path.join(OUTPUT_DIR, "loss_plot.png")
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.savefig(loss_plot_path)
        plt.close()

        model_path = os.path.join(OUTPUT_DIR, "trained_model.pt")
        torch.save(model, model_path)

        yield loss_plot_path, None, model_path, None, "Training Complete"

    except Exception as e:
        print(f"Error during training: {e}")
        yield None, None, None, None, f"Error: {str(e)}"

def train_model_with_default_path(loss_name, opt_name, lr, batch_size, image_size,
                                  file, custom_path, epochs, num_channels, 
                                  generate_animation, target_frames, frame_rate):
    if not custom_path or custom_path.strip() == "":
        custom_path = os.path.join(os.path.expanduser("~"), "my_neuralnt_data")
        os.makedirs(custom_path, exist_ok=True)

    return train_model(loss_name, opt_name, lr, batch_size, image_size, file, custom_path, epochs, num_channels, generate_animation, target_frames, frame_rate)
