# What is this project ?

This code is for solving the Shortest Vector Problem (SVP) while using the a simulated Coherent Ising Machine (CIM) and a Coherent Potts Machine (CPM) as a subroutine; the code for the CIM simulator is publicly available at https://github.com/mcmahon-lab/cim-optimizer. 

It is part of my MSc Thesis at the Imperial College London.

Feel free to give me feedback !

# Project Requirements

## Main Required Packages

- [`cim-optimizer`](https://github.com/mcmahon-lab/cim-optimizer): Installed directly from GitHub. Used for Coherent Ising Machine optimization.
- [`BOHB-HPO`](https://pypi.org/project/BOHB-HPO/): Bayesian Optimization and Hyperband for efficient hyperparameter tuning.
- [`PyTorch`](https://pytorch.org/): Deep learning framework. Includes:
  - `torch`
  - `torchvision`
  - `torchaudio`
- [`fpylll`](https://github.com/fplll/fpylll): Library for lattice algorithms such as the LLL algorithm.
- `numpy`: For numerical computations.
- `matplotlib`: For plotting and visualizations.

Check the full requirements in the pyproject file ...

## Installation

From the project root: 

Also, make sure to clone the cim-optimiser GIT and add there the scripts stored in the "CPM" folder of this project. This is to enable CPM simulation. 

## Environment Notes

Python 3.10 with virtual enviromnent used. 
Hardware: Mac M1 silicon (16 GB, 8 cores)
