# gym-swirl
## About
`gym-swirl` is a software package for simulating active colloids with social interaction. The simulation is a python implementation of the article "[Formation of stable and responsive collective states in suspensions of active colloids](https://www.nature.com/articles/s41467-020-16161-4)" by Bäuerle et al. It is written as an [OpenAI Gym](http://gym.openai.com) package using [PyTorch](https://pytorch.org) to better suit the needs for a reinforcement learning model.

## Installation
The software package is tested with python version `>=3.8`.
Using [`pip`](https://packaging.python.org/tutorials/installing-packages/), the package is installed by running the following command in the project root folder:

```
pip install -e .
```

To avoid dependency conflicts, using [conda](https://docs.conda.io/en/latest/) or other python environment and package management systems is recommended.

## Usage
The simulation can easily be run and displayed using a [jupyter notebook](https://jupyter.org). For an example usage, see [`notebooks/simulation.ipynb`](notebooks/simulation.ipynb).

The default arguments, that can be overwritten by passing them into the `env.reset()` as keyword arguments, are as follows:
```python
defaults = {
	"amount": 48, # number of particles
	"spread": 5*6.3e-6, # the standard deviation of the initial position in m
	"T": 0., # starting time in s
	"T_release": 200., # release time in s
	"velocity": 0.5e-6, # velocity magnitude in m/s
	"diameter": 6.3e-6, # particle diameter in m
	"Rc": 6.3e-6/2, # radius of collision in m
	"Rr": 8e-6, # radius of repulsion in m
	"Ro": 25e-6, # radius of orientation in m
	"Ra": float("inf"), # radius of attraction in m
	"alpha": 2. * pi, # visibility angle in rad
	"DT": 0.014e-12, # translational diffusion constant in m^2 s^−1
	"DR": 0.0028, # rotational diffusion constant in s^−1
	"Gamma": 25., # max torque factor (unitless)
	"dt": 0.2, # time step size in s
	"random_angles": True, # boolean for initial random angles
}
```

## Credits
Author [Johan Dettmar](mailto:dettmar@gmail.com) 2020-2021.
