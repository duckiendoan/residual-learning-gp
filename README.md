# Velocity-Dependent Residual Learning in Nonlinear Systems
*Project done as part of the minicourse Gaussian Processes for Learning, Modelling, and Control of Mechanical Systems by Dr. Leonardo Colombo at University of Verona.*

A study of whether velocity information is necessary in residual learning on nonlinear systems using Gaussian Processes.

## Experiments
- **Interpolation**: Train a Gaussian Process on slow/fast system, test on the same system. Data simulated using Computed Torque Control.
- **Extrapolation**: Train on slow system (low velocity), test on fast system.
- **Evaluation**: 1-step RMSE and closed-loop tracking.

## Code structure:
- `simulation.py`: Code for robotic simulation (a two-link planar robot arm)
- `models.py`: Machine Learning models: Gaussian Process, Gaussian Process with physics-informed features.
- `main.py`: The main experiment script:
  - `interpolation_exp`: Interpolation experiment.
  - `extrapolation_exp_fail`: A first failed attempt at extrapolation.
  - `extrapolation_exp_successful`: A successful attempt at extrapolation (with physics-informed learning).

## Installation
```bash
pip install numpy matplotlib scikit-learn
```
