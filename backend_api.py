import os
import shutil
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from torchvision import transforms
import uvicorn
import json

from layers import add_layer, update_layer, delete_layer, reset_layers, layer_configs, update_architecture_text
from training import train_model_with_default_path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LayerInput(BaseModel):
    layer_type: str
    in_dim: Optional[str] = ""
    out_dim: Optional[str] = ""

@app.get("/architecture")
def get_architecture():
    return {"text": update_architecture_text(), "layers": layer_configs}

@app.post("/add_layer")
def api_add_layer(layer: LayerInput):
    res = add_layer(layer.layer_type, layer.in_dim, layer.out_dim)
    return {"architecture": res}

@app.post("/reset")
def api_reset():
    reset_layers()
    return {"status": "reset"}

@app.post("/train")
async def api_train(
    loss_name: str = Form(...),
    opt_name: str = Form(...),
    lr: str = Form(...),
    batch_size: str = Form(...),
    image_size: str = Form(...),
    epochs: str = Form(...),
    num_channels: int = Form(...),
    dataset: UploadFile = File(...)
):
    temp_file = f"temp_{dataset.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(dataset.file, buffer)

    try:
        results = None
        for update in train_model_with_default_path(
            loss_name, opt_name, lr, batch_size, image_size,
            temp_file, "", epochs, num_channels,
            False, "300", "10"
        ):
            results = update

        return {
            "loss_plot": results[0] if results else None,
            "logs": results[4] if results else "Training completed"
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/predict")
async def api_predict(image: UploadFile = File(...), image_size: int = Form(32)):
    model_path = os.path.join("outputs", "trained_model.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="No trained model found. Train first!")

    try:
        # Load the model - FIX: Added weights_only=False to support full model loading in newer Torch versions
        model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()

        # Prepare the image
        img = Image.open(image.file).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, class_idx = torch.max(probabilities, 0)

        # CIFAR-10 labels
        labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

        return {
            "prediction": labels[class_idx.item()] if class_idx.item() < len(labels) else f"Class {class_idx.item()}",
            "confidence": f"{confidence.item() * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
