import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
from datetime import datetime
from scipy.spatial.distance import squareform
from .State import State

cossim = nn.CosineSimilarity(dim=1, eps=1e-14)
pi = 3.1415927410125732

defaults = {
	"Delta": 67.5 * pi / 180, # rad
	"T": 0., # s
	"T_release": 200., # s
	"amount": 48,
	"velocity": 0.5e-6, # m/s
	"diameter": 6.3e-6, # m
	"Rc": 6.3e-6/2, # m
	"Rr": 8e-6, # m
	"Ro": 25e-6, # m
	"Ra": float("inf"), # m
	"alpha": 225. * pi / 180, # rad
	"DT": 0.014e-12, # m^2 s^−1
	"DR": 0.0028, # s^−1
	"Gamma": 25., # unitless
	"dt": 0.2, # s
	"random_angles": True,
}

class ActiveParticles(nn.Module):

	def __init__(self, amount=50, spread=2, **kwargs):
		super(ActiveParticles, self).__init__()

		self.__dict__.update((k, torch.tensor(v)) for k, v in { **defaults, **kwargs }.items() if k in defaults)
		self.positions = torch.normal(mean=0, std=self.diameter*spread, size=(self.amount,), dtype=torch.cfloat)

		if self.random_angles:
			self.orientations = torch.normal(mean=0, std=self.diameter*spread, size=(self.amount,), dtype=torch.cfloat)
			self.orientations /= self.orientations.abs()
		else:
			self.orientations = self.positions.mean()-self.positions
			self.orientations /= self.orientations.abs()
			self.orientations *= torch.exp(Delta*1j)

			rotational_noise = torch.normal(mean=0, std=torch.sqrt(torch.tensor(2*DR)), size=(self.amount,))
			self.orientations *= torch.exp((rotational_noise*torch.sqrt(torch.tensor(dt)))*1.j)



	def save(self, basename="ap_states"):
		# TODO: store all settings
		filename = f"runs/{basename}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pkl"
		print(f"Storing progress as {filename}")
		with open(filename, "wb") as f:
			pickle.dump(self.states, f)

		return filename


	def load(self, filename, restart_from_id=-1, delete_earlier_states=False):
		# TODO: load all settings from here
		print(f"Loading progress as {filename}")
		with open(filename, "rb") as f:
			print("self", self)
			self.states = pickle.load(f)
			self.restart_from(restart_from_id, delete_earlier_states=delete_earlier_states)
			print(f"Amount {self.amount}, T {self.T}, steps done: {len(self.states)}")


	def timesteps(self, steps, betweensteps=1):
		""" Timestep takes several steps forward,
			with a specified amount of steps inbetween
		"""
		for step in range(steps):
			self.timestep(steps=betweensteps)


	def timestep(self, steps=1):

		for step in range(steps):

			self.T += self.dt
			translation, rotation, orientation_sums, leftturns, rightturns = self.move()
			self.positions += translation
			self.orientations *= rotation
			self.solve_collisions()

		state_values = [
			self.positions,
			self.orientations,
			self.dt,
			self.T,
			self.O_R().mean(),
			self.O_P(),
			self.Delta,
			self.DT,
			self.DR,
			self.Gamma,
			orientation_sums,
			leftturns,
			rightturns
		]

		return State(*[x.detach().clone() for x in state_values])


	@staticmethod
	def getdistances(positions, is_complex=False):
		if is_complex:
			return (positions.repeat(len(positions), 1).T - positions).T
		else:
			absdists = F.pdist(torch.view_as_real(positions))
			return torch.tensor(squareform(absdists))


	@staticmethod
	def get_anglediff(a, b):
		aang = a.angle()
		bang = b.angle()

		diff = aang-bang
		diff = torch.where(diff <= -pi, diff % pi, diff) # if below -180deg convert it to positive equiv
		diff -= (diff >= pi) * pi * 2 # if above 180 deg remove 2 pi

		return diff

	def move(self):

		avoid_collisions = False

		with torch.no_grad():

			alldists = ActiveParticles.getdistances(self.positions)
			# TODO: clean up these matrices
			inside_Rr = (alldists <= self.Rr).type(torch.cfloat)
			inside_Rr -= torch.eye(self.amount, dtype=torch.cfloat) # remove own particle
			inside_Ro = (alldists <= self.Ro+self.Rc).type(torch.cfloat)
			inside_Ra = (alldists <= self.Ra+self.Rc).type(torch.cfloat)

			if self.alpha < 2 * pi:
				abs_angles_diff = ActiveParticles.get_anglediff(self.orientations.repeat(self.amount, 1).T, self.orientations).abs()
				within_view = (abs_angles_diff < self.alpha).type(torch.cfloat)
				in_front = (abs_angles_diff < pi/2).type(torch.cfloat)
				inside_Ra = torch.where(inside_Ra.real.type(torch.bool), within_view, torch.tensor([0.+0.j]))
				inside_Ro = torch.where(inside_Ro.real.type(torch.bool), within_view, torch.tensor([0.+0.j]))
				inside_Rr = torch.where(inside_Rr.real.type(torch.bool), in_front, torch.tensor([0.+0.j]))

			# repulsion
			n_r = inside_Rr.real.sum(axis=1)
			S = torch.mv(inside_Rr, self.positions) / torch.max(n_r, torch.tensor([1.])) - self.positions*n_r.sign() # only calc number if has neightbours
			d = -S

			# attraction
			n_a = inside_Ra.real.sum(axis=1)
			cms = torch.mv(inside_Ra, self.positions) / torch.max(n_a, torch.tensor([1.]))
			Ps = cms - self.positions*n_a.sign()

			# orientation
			orientation_sums = torch.mv(inside_Ro, self.orientations)

			leftturns = Ps*torch.exp(self.Delta*1j)
			rightturns = Ps*torch.exp(-self.Delta*1j)
			if self.T >= self.T_release:
				left_closer = cossim(torch.view_as_real(leftturns), torch.view_as_real(orientation_sums)) >= cossim(torch.view_as_real(rightturns), torch.view_as_real(orientation_sums))
				best_turns = torch.where(left_closer, leftturns, rightturns)
			else:
				best_turns = leftturns

			# find if it's to the right or left of current direction
			# if right, go left vice versa
			rotational_noise = torch.normal(mean=0., std=1., size=(self.amount,)) * torch.sqrt(2*self.DR)

			angle_to_target = torch.where(d.abs() > 0.,
				ActiveParticles.get_anglediff(d, self.orientations),
				ActiveParticles.get_anglediff(best_turns, self.orientations))

			if avoid_collisions:

				current_angles = self.orientations.angle()
				current_angles += (current_angles < 0) * 2 * pi

				inside_Avoidance = (alldists <= 3*self.Rc).type(torch.float) - torch.eye(self.amount)
				# find the common center between two particles
				inside_Avoidance_compl = ActiveParticles.getdistances(self.positions, True)#.angle()
				take_left = ActiveParticles.get_anglediff(self.orientations.repeat(self.amount, 1).T, inside_Avoidance_compl) > 0.
				assert take_left.shape == (self.amount, self.amount)
				avoidance_rotation_angle = ((take_left*2-1) * inside_Avoidance).sum(dim=1).sign() * pi/2.
				assert avoidance_rotation_angle.shape == (self.amount,), avoidance_rotation_angle.shape
				angle_to_target += avoidance_rotation_angle

			rotation = torch.exp((self.dt * self.Gamma * self.DR * torch.sin(angle_to_target) + rotational_noise*torch.sqrt(self.dt))*1.j)

			# translation
			translational_noise = torch.normal(mean=0., std=1., size=(self.amount,), dtype=torch.cfloat) * torch.sqrt(2*self.DT)
			translation = self.dt * self.velocity * self.orientations + translational_noise * torch.sqrt(self.dt)

		return translation, rotation, orientation_sums, leftturns, rightturns


	def solve_collisions(self, depth=0):

		alldists_compl = ActiveParticles.getdistances(self.positions, True)
		alldists_abs = alldists_compl.abs()
		all_collisions = alldists_abs <= 2*self.Rc - torch.eye(self.amount)
		overlap_distance = 2.1*self.Rc - alldists_abs
		# TODO why isnt this breaking for alldists_abs == 0?
		move_distance = torch.where(all_collisions, (alldists_compl/alldists_abs) * (overlap_distance/2), torch.tensor([0.+0.j]))
		self.positions -= move_distance.sum(dim=1)

		if all_collisions.sum() and depth < 1000:
			self.solve_collisions(depth + 1)


	def O_R(self):

		cm = self.positions.mean()
		r = self.positions-cm
		r /= r.abs()
		r = F.pad(torch.view_as_real(r), pad=(0,1,0,0), mode="constant", value=0)
		u = self.orientations / self.orientations.abs()
		u = F.pad(torch.view_as_real(u), pad=(0,1,0,0), mode="constant", value=0)
		e_z = torch.tensor([0,0,1], dtype=torch.float)

		return torch.mv(torch.cross(r, u), e_z)


	def local_O_R(self, measurement_positions):

		dists = torch.cdist(measurement_positions.reshape((-1,2)), torch.view_as_real(self.positions))
		exp_dists = torch.exp(-torch.abs(dists)**2/(2*self.diameter**2))

		return torch.mv(exp_dists, self.O_R()) / (exp_dists.sum(axis=1) + 1e-14)


	def O_P(self):

		return self.orientations.sum(axis=0).abs().sum() / self.amount
