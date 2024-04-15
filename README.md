# Physics-Informed Neural Networks for PDEs

## Overview
This repository contains implementations of Physics-Informed Neural Networks (PINNs) designed to solve various Partial Differential Equations (PDEs). PINNs are a novel deep learning framework that incorporates the underlying physics of the problem, expressed by PDEs, into the neural network. This repository focuses on famous PDEs and demonstrates their implementation in Pytorch.

## Contents
The repository includes the following Python scripts:
- `2d-heat-equation.py`: PINN solver for the 2D Heat Equation, a parabolic PDE modeling heat distribution.
- `2d-poisson-equation.py`: PINN approach to the 2D Poisson Equation, an elliptic PDE for potential fields.
- `2d-wave-equation.py`: Applies PINNs to the 2D Wave Equation, a hyperbolic PDE for wave propagation.
- `burgers-equation.py`: Solves the Burgers' Equation (included as an example of a non-linear PDE).
- `helmholtz-equation.py`: Solves the Helmholtz Equation (to demonstrate PINNs on an eigenvalue problem).
- `ns-equation.py`: Solves the Navier-Stokes Equation (to show applicability to fluid dynamics).

## Installation

To run these scripts, ensure you have the following prerequisites installed:

```bash
pip install numpy torch
```

Clone the repository to your local machine:

```bash
git clone https://github.com/tavakkoliamirmohammad/pinn-pde-solver.git
cd pinn-pde-solver
```

## Usage

To solve a PDE using a PINN, navigate to the cloned directory and run the corresponding script. For example:

```bash
python 2d-heat-equation.py
```


## Contributing
We welcome contributions to this repository. If you would like to add a new PDE solver or improve the existing ones, please fork the repository and submit a pull request.

## License
[MIT License](LICENSE)

---