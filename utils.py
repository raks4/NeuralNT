import os
import tempfile

def safe_output(path):
    if os.path.isdir(path):
        print(f"Warning: Expected a file, got a directory: {path}")
    return path

def parse_int_or_tuple(val):
    try:
        if ',' in str(val):
            return tuple(map(int, str(val).split(',')))
        return int(val)
    except Exception:
        print(f"Warning: Invalid numeric input: '{val}'. Defaulting to 1.")
        return 1

def get_default_writable_folder():
    home_dir = os.path.expanduser("~")
    default_path = os.path.join(home_dir, "my_gradio_data")
    os.makedirs(default_path, exist_ok=True)
    return default_path
