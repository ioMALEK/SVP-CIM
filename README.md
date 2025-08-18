# What is this project ?

This code is for solving the Shortest Vector Problem (SVP) while using the a simulated Coherent Ising Machine (CIM) as a subroutine; the code for the CIM simulator is publicly available at https://github.com/mcmahon-lab/cim-optimizer. 

It is part of my MSc Thesis at the Imperial College London, available at [TO BE UPLOADED ON THE ARXIV]

Feel free to give me feedback !

# Project Requirements

This project uses a combination of quantum-inspired optimization tools, deep learning frameworks, and numerical libraries.

## Required Packages

- [`cim-optimizer`](https://github.com/mcmahon-lab/cim-optimizer): Installed directly from GitHub. Used for Coherent Ising Machine optimization.
- [`BOHB-HPO`](https://pypi.org/project/BOHB-HPO/): Bayesian Optimization and Hyperband for efficient hyperparameter tuning.
- [`PyTorch`](https://pytorch.org/): Deep learning framework. Includes:
  - `torch`
  - `torchvision`
  - `torchaudio`
- [`fpylll`](https://github.com/fplll/fpylll): Library for lattice algorithms such as the LLL algorithm.
- `numpy`: For numerical computations.
- `matplotlib`: For plotting and visualizations.

## Hardware Notes

- If you're using a GPU:
  - For NVIDIA GPUs, install the appropriate CUDA-enabled PyTorch version from [pytorch.org](https://pytorch.org/get-started/locally/).
  - For Apple Silicon (M1/M2/M3), install the PyTorch Metal backend:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/metal
    ```

## Installation

Install all Python dependencies with:

```bash
pip install -r requirements.txt
