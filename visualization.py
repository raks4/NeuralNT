import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from sklearn.decomposition import PCA
import tempfile
import torch

# --------------------------------
# Graph Functions
# --------------------------------

def get_flat_weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def generate_loss_plot(loss_history):
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name)
    tmpfile.close()
    plt.close(fig)
    return tmpfile.name

def generate_3d_animation_pca(weight_path, loss_history, output_path, target_frames=300, frame_rate=10):
    if len(loss_history) == 0 or np.any(np.isnan(loss_history)):
        print("Warning: loss_history is empty or contains NaNs — cannot animate.")
        return

    if len(weight_path) < target_frames:
        target_frames = len(weight_path)

    indices = np.linspace(0, len(weight_path) - 1, target_frames).astype(int)
    weight_path_sampled = weight_path[indices]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(weight_path_sampled)

    interpolated_loss = np.interp(
        np.linspace(0, len(loss_history) - 1, target_frames),
        np.arange(len(loss_history)),
        loss_history
    )

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Training Progression")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("Loss")

    ax.plot(reduced[:, 0], reduced[:, 1], interpolated_loss, color='red', alpha=0.6)
    point, = ax.plot([reduced[0, 0]], [reduced[0, 1]], [interpolated_loss[0]], 'ro')

    def update(i):
        point.set_data([reduced[i, 0]], [reduced[i, 1]])
        point.set_3d_properties([interpolated_loss[i]])
        return (point,)

    interval = 1000 / frame_rate  
    ani = animation.FuncAnimation(
        fig, update,
        frames=target_frames,
        interval=interval,
        blit=False
    )

    writer = FFMpegWriter(
        fps=frame_rate,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p"]
    )

    ani.save(output_path, writer=writer)
    plt.close(fig)
    return output_path
