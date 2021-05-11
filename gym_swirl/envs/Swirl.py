import os
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

		# State is immutable and _replace returns a copy of last state
		current_state = self.states[-1]._replace(Deltas=self.states[-1].Deltas + actions)
		states = self.aps.timesteps(current_state, steps=steps, betweensteps=betweensteps)
		self.states += states


	def reset(self, **kwargs):

		self.settings = { **defaults, **kwargs }

		positions = torch.normal(mean=0, std=self.settings["spread"], size=(self.settings["amount"],), dtype=torch.cfloat)
		orientations = torch.normal(mean=0, std=1., size=(self.settings["amount"],), dtype=torch.cfloat)
		orientations /= orientations.abs()
		Deltas = torch.tensor(self.settings["Deltas"])

		self.aps = ActiveParticles(**kwargs)
		self.states = [self.aps.state(positions, orientations, Deltas)]


	def render(self, filename=None, **kwargs):
		anim = Animator(self, **kwargs)
		if filename is not None:
			return anim.store(filename)
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
		if len(basename.split("/")) >= 2 and not os.path.isdir(basename.split("/")[0]):
			os.mkdir(basename.split("/")[0])
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
			self.Deltas = self.states[-1].Deltas # torch.ones(self.amount) *
			print(f"Amount particles {len(self.states[-1].positions)}, amount states: {len(self.states)}")


	def close(self):
		pass
