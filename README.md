# Velocity-Dependent Residual Learning in Nonlinear Systems
*Project done as part of the minicourse Gaussian Processes for Learning, Modelling, and Control of Mechanical Systems by Dr. Leonardo Colombo at University of Verona.*

A study of whether velocity information is necessary in residual learning on nonlinear systems using Gaussian Processes.

## Experiments
- **Interpolation**: Train a Gaussian Process on slow/fast system, test on the same system. Data simulated using Computed Torque Control
- **Extrapolation**: Train on slow system (low velocity), test on fast system.
- **Evaluation**: 1-step RMSE and closed-loop tracking.

## Installation
```bash
pip install numpy matplotlib scikit-learn
```
