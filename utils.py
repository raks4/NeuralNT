import os
import tempfile
import gradio as gr

def safe_output(path):
    if os.path.isdir(path):
        gr.Warning(f"Expected a file, got a directory: {path}")
    return path

def parse_int_or_tuple(val):
    try:
        return tuple(map(int, str(val).split(','))) if ',' in str(val) else int(val)
    except Exception:
        gr.Warning(f"Invalid numeric input: '{val}'. Please enter an integer or comma-separated pair.")


# This function gets a subfolder in the home directory which is not protected and is writeable
# Input: Nothing
# Output: A writeable folder
def get_default_writable_folder():

    home_dir = os.path.expanduser("~")  # e.g. C:\\Users\\Alice on Windows
    default_path = os.path.join(home_dir, "my_gradio_data")
    os.makedirs(default_path, exist_ok=True)
    return default_path