import os
import zipfile
import tempfile
import shutil
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# --------------------------------
# Data Loaders
# --------------------------------

def extract_zip_to_tempdir(file_like, custom_path=None):
    if custom_path:
        os.makedirs(custom_path, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=custom_path)
    else:
        temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(file_like, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_pil_loader(path, num_channels=3):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB' if num_channels == 3 else 'L')
            img.load()
            return img
    except Exception as e:
        print(f"Skipping corrupted image: {path} — {e}")
        return None

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, num_channels=3):
        loader_with_channels = lambda path: safe_pil_loader(path, num_channels)
        super().__init__(root, transform=transform, loader=loader_with_channels)
        
        self.samples = [s for s in self.samples if loader_with_channels(s[0]) is not None]
        self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if img is None:
            print(f"Warning: Could not load image at {path}")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def load_data(file, custom_path=None, batch_size=32, image_size=28, num_channels=3, loss_fn=None):
    ext = os.path.splitext(file)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file)
        if 'y' not in df.columns:
            print("Warning: CSV must contain a 'y' column.")
        
        X = df.drop(columns=['y']).values.astype(np.float32)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            y = df['y'].values.astype(np.int64)
        elif isinstance(loss_fn, (nn.MSELoss, nn.BCEWithLogitsLoss)):
            y = df['y'].values.astype(np.float32)
        else:
            print(f"Warning: Unhandled loss function type: {type(loss_fn)}")
    
        X, y = shuffle(X, y)

        return {
            "type": "tabular",
            "train": (X, y),
            "path": None
        }

    elif ext == ".zip":
        with open(file, 'rb') as f:
            data_dir = extract_zip_to_tempdir(f, custom_path)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        dataset = SafeImageFolder(data_dir, transform=transform, num_channels=num_channels)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return {
            "type": "image",
            "train": train_loader,
            "path": data_dir
        }
    else:
        print("Warning: Unsupported file format. Provide a .csv or .zip path.")
