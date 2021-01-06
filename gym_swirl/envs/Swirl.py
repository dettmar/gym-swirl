import gym
from gym import error, spaces, utils
from gym.utils import seeding
import torch
#from torchdiffeq import odeint_adjoint as odeint

from .ActiveParticles import ActiveParticles
from ..animator import Animator

defaults = {
	"T": 0., # s
	"amount": 48,
	"spread": 5*6.3e-6,
}


class Swirl(gym.Env):

	metadata = { "render.modes": ["human"] }

	def __init__(self):
		#self.reset()
		pass


	def step(self, actions, steps=10, betweensteps=1):

		self.deltas += actions
		states = self.aps.timesteps(self.positions, self.orientations, self.deltas, steps=steps, betweensteps=betweensteps)
		self.states += states


	def reset(self, **kwargs):

		settings = { **defaults, **kwargs }
		self.amount = settings["amount"]
		self.positions = torch.normal(mean=0, std=settings["spread"], size=(self.amount,), dtype=torch.cfloat)
		self.orientations = torch.normal(mean=0, std=1., size=(self.amount,), dtype=torch.cfloat)
		self.orientations /= self.orientations.abs()
		self.deltas = torch.tensor(settings["Delta"]) # torch.ones(self.amount) *
		self.aps = ActiveParticles(**kwargs)
		self.states = []


	def render(self, **kwargs):
		anim = Animator(self, **kwargs)
		return anim.show()


	def close(self):
		pass
