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

def _download_file(url, save_dir="downloads", filename=None):
    """Download a file using aria2 and return the path to the downloaded file."""
    print(f"Downloading {url}...")
    
    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Use custom filename or extract from URL
    filename = filename or url.split("/")[-1]
    full_path = save_path / filename

    # Run aria2c download command
    result = subprocess.run(
        ["aria2c", "-q", "-x", "16", "-d", str(save_path), "-o", filename, url],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download {url}. Error: {result.stderr.decode()}")

    print(f"Downloaded to {full_path}")
    return full_path


def _load_model(model_name):
    install_aria2()
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="meta", mmap=True)

    else:  # load from hub
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        # url = "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt"
        downloaded_file = _download_file(url)
        print(f"Loading downloaded model: {downloaded_file}")
        # torch.serialization.add_safe_globals([str(downloaded_file)])
        model_data = torch.load(str(downloaded_file), map_location="cpu", mmap=True)
    '''
    JOJO: cfg and model_state are the most important information got from pretrained model.
    Get all the parameters needed for folding trunk (like everything shown in trunk.py Folding
    Trunk Config and Structure Module Config)
    '''
    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    '''
    JOJO: Here is the place ESMFold is loaded.
    '''
    model = ESMFold(esmfold_config=cfg)
    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)
    #JO: These functions keys will not be used in the model, so it's fine to ignore them
    # if missing_essential_keys:
    #     raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")
    
    '''
    JOJO: At this point, all the weights and bias are loaded into the model.
    Then it's ready to use
    '''
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
