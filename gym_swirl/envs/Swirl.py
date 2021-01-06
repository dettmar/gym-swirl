import datetime
import pickle
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import torch
#from torchdiffeq import odeint_adjoint as odeint

from .ActiveParticles import ActiveParticles
from ..animator import Animator

defaults = {
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

		self.settings = { **defaults, **kwargs }
		self.amount = self.settings["amount"]
		self.positions = torch.normal(mean=0, std=self.settings["spread"], size=(self.amount,), dtype=torch.cfloat)
		self.orientations = torch.normal(mean=0, std=1., size=(self.amount,), dtype=torch.cfloat)
		self.orientations /= self.orientations.abs()
		self.deltas = torch.tensor(self.settings["Delta"]) # torch.ones(self.amount) *
		self.aps = ActiveParticles(**kwargs)
		self.states = []


	def render(self, **kwargs):
		anim = Animator(self, **kwargs)
		return anim.show()


	def save(self, basename="runs/run"):
		# TODO: store all settings
		date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		filename = f"{basename}_{date}.pkl"
		print(f"Storing progress as {filename}")
		data = {
			"settings": self.settings,
			"states": self.states
		}
		with open(filename, "wb") as f:
			pickle.dump(data, f)

		return filename


	def load(self, filename, restart_from_id=-1, delete_earlier_states=False):
		# TODO: load all settings from here
		print(f"Loading progress as {filename}")
		with open(filename, "rb") as f:
			data = pickle.load(f)
			data["settings"]["T"] = data["states"][-1].T
			self.reset(**data["settings"])

			self.states = data["states"]
			self.positions = self.states[-1].positions
			self.orientations = self.states[-1].orientations
			self.deltas = self.states[-1].Delta # torch.ones(self.amount) *
			print(f"Amount particles {self.amount}, amount states: {len(self.states)}")

	def close(self):
		pass
