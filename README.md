# Class Distribution Shifts in Zero-Shot Learning: Learning Robust Representations

This repository accompanies the paper
Class Distribution Shifts in Zero-Shot Learning: Learning Robust Representations.

#### Requirements
The file `requirements.txt` lists the package requirements. 

For convenience the notebooks were adapted for running also in Google Colab. 
_______________________________________________________

##### Experiments
- `Simulations.ipynb` - A single repetition of a simulation with synthetic data.
- `CelebA dataset.ipynb` - A single repetition of the face recognition experiment.
- `ETHEC dataset.ipynb` - A single repetition of the species recognition experiment.

All notebooks use the file `algorithm.py`, that includes the training function and functions to calculate penalties.
The file `pairs.py` includes helper functions to perform hierarchical sampling.
The `Simulations.ipynb` notebook uses `synthetic_data.py` for data generation.

