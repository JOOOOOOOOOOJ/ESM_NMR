from pathlib import Path

import torch

from esm.esmfold.v1.esmfold import ESMFold

import os
import subprocess

def install_aria2():
    """Ensure aria2 is installed."""
    try:
        subprocess.run(["aria2c", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Installing aria2...")
        os.system("apt-get install aria2 -qq")

def _download_file(url):
    """Download a file using aria2."""
    print(f"Downloading {url}...")
    result = subprocess.run(
        ["aria2c", "-q", "-x", "16", url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download {url}. Error: {result.stderr.decode()}")

def _load_model(model_name):
    install_aria2()
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
    else:  # load from hub
        print('nihao')
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        _download_file(url)
    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model


def esmfold_v0():
    """
    ESMFold v0 model with 3B ESM-2, 48 folding blocks.
    This version was used for the paper (Lin et al, 2022). It was trained 
    on all PDB chains until 2020-05, to ensure temporal holdout with CASP14
    and the CAMEO validation and test set reported there.
    """
    return _load_model("esmfold_3B_v0")


def esmfold_v1():
    """
    ESMFold v1 model using 3B ESM-2, 48 folding blocks.
    ESMFold provides fast high accuracy atomic level structure prediction
    directly from the individual sequence of a protein. ESMFold uses the ESM2
    protein language model to extract meaningful representations from the
    protein sequence.
    """
    return _load_model("esmfold_3B_v1")
