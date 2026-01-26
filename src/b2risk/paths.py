from __future__ import annotations
import os

def repo_root() -> str:
    # Works in Colab after `cd /content/b2-tokenized-treasury`
    return os.getcwd()

def processed_dir(root: str | None = None) -> str:
    root = root or repo_root()
    return os.path.join(root, "data", "processed")

def ensure_dirs(root: str | None = None) -> None:
    root = root or repo_root()
    os.makedirs(os.path.join(root, "data", "processed"), exist_ok=True)
