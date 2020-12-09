import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .ActiveParticles import ActiveParticles




class Swirl(gym.Env):

	metadata = { "render.modes": ["human"] }

	def __init__(self):
		#self.reset()
		pass


	def step(self, action):
		return self.aps.timestep(1)


	def reset(self, **kwargs):
		self.aps = ActiveParticles(**kwargs)


	def render(self, mode="human"):
		pass


	def close(self):
		pass
