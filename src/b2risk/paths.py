import os

def get_project_root() -> str:
    # Assumes notebooks/ is one level below repo root
    # If running from repo root, this still works if you set root manually.
    return os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()

def get_data_dirs(root: str):
    raw_dir = os.path.join(root, "data", "raw")
    proc_dir = os.path.join(root, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    return raw_dir, proc_dir
