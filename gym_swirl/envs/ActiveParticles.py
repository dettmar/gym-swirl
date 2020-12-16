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
	"T": 0., # s
	"T_release": 200., # s
	"velocity": 0.5e-6, # m/s
	"diameter": 6.3e-6, # m
	"Rc": 6.3e-6/2, # m
	"Rr": 8e-6, # m
	"Ro": 25e-6, # m
	"Ra": float("inf"), # m
	"alpha": 360. * pi / 180, # rad
	"DT": 0.014e-12, # m^2 s^−1
	"DR": 0.0028, # s^−1
	"Gamma": 25., # unitless
	"dt": 0.2, # s
	"random_angles": True,
}

class ActiveParticles(nn.Module):

	def __init__(self, **kwargs):

		super(ActiveParticles, self).__init__()

		settings = { **defaults, **kwargs }
		self.__dict__.update((k, torch.tensor(v)) for k, v in settings.items() if k in defaults)


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
			print(f"T {self.T}, steps done: {len(self.states)}")


	def timesteps(self, positions, orientations, deltas, steps, betweensteps=1):
		""" Timestep takes several steps forward,
			with a specified amount of steps inbetween
		"""
		states = []
		for step in range(steps):
			states.append(self.timestep(positions, orientations, deltas, steps=betweensteps))

		return states


	def timestep(self, positions, orientations, deltas, steps=1):

		for step in range(steps):

			self.T += self.dt
			positions, orientations, orientation_sums, leftturns, rightturns = self.forward(positions, orientations, deltas)

		if deltas.numel() > 1:
			ORs = self.local_O_R(positions, positions, orientations)
		else:
			ORs = self.O_R(positions, orientations).mean()

		state_values = [
			positions,
			orientations,
			self.dt,
			self.T,
			ORs,
			self.O_P(orientations),
			deltas,
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
		"""Takes a complex vector of n particles and calculates
		a distance matrix for all inbetween distances.
		"""
		if is_complex:
			return (positions.repeat(len(positions), 1).T - positions).T
		else:
			absdists = F.pdist(torch.view_as_real(positions))
			return torch.tensor(squareform(absdists))


	@staticmethod
	def get_anglediff(a, b):
		"""Takes two complex vectors and calculates the (positive or negative)
		angle between them.
		"""
		aang = a.angle()
		bang = b.angle()

		diff = aang-bang
		diff = torch.where(diff <= -pi, diff % pi, diff) # if below -180deg convert it to positive equiv
		diff -= (diff >= pi) * pi * 2 # if above 180 deg remove 2 pi

		return diff

	def forward(self, positions, orientations, deltas):
		"""Translates and rotates all particles based on three rules:
		repulsion, attraction and orientation.
		"""
		#positions, orientations, deltas, actions = torch.split(x, (1,1,1,1))
		amount = len(positions)

		with torch.no_grad():

			alldists = ActiveParticles.getdistances(positions)
			# TODO: clean up these matrices
			inside_Rr = (alldists <= self.Rr).type(torch.cfloat)
			inside_Rr -= torch.eye(amount, dtype=torch.cfloat) # remove own particle
			inside_Ro = (alldists <= self.Ro+self.Rc).type(torch.cfloat)
			inside_Ra = (alldists <= self.Ra+self.Rc).type(torch.cfloat)

			abs_angles_diff = ActiveParticles.get_anglediff(orientations.repeat(amount, 1).T, orientations).abs()
			in_front = (abs_angles_diff < pi/2).type(torch.cfloat)
			inside_Rr = torch.where(inside_Rr.real.type(torch.bool), in_front, torch.tensor([0.+0.j]))

			if self.alpha < 2 * pi:
				within_view = (abs_angles_diff < self.alpha).type(torch.cfloat)
				inside_Ra = torch.where(inside_Ra.real.type(torch.bool), within_view, torch.tensor([0.+0.j]))
				inside_Ro = torch.where(inside_Ro.real.type(torch.bool), within_view, torch.tensor([0.+0.j]))

			# repulsion
			n_r = inside_Rr.real.sum(axis=1)
			S = torch.mv(inside_Rr, positions) / torch.max(n_r, torch.tensor([1.])) - positions * n_r.sign() # only calc distance if neightbours exist
			d = -S

			# attraction
			n_a = inside_Ra.real.sum(axis=1)
			cms = torch.mv(inside_Ra, positions) / torch.max(n_a, torch.tensor([1.]))
			Ps = cms - positions * n_a.sign()

			# orientation
			orientation_sums = torch.mv(inside_Ro, orientations)

			# select best side
			leftturns = Ps * torch.exp(deltas * 1j)
			rightturns = Ps * torch.exp(-deltas * 1j)
			if self.T >= self.T_release:
				left_closer = cossim(*map(torch.view_as_real, [leftturns, orientation_sums])) >= cossim(*map(torch.view_as_real, [rightturns, orientation_sums]))
				best_turns = torch.where(left_closer, leftturns, rightturns)
			else:
				best_turns = leftturns

			angle_to_target = torch.where(d.abs() > 0.,
				ActiveParticles.get_anglediff(d, orientations),
				ActiveParticles.get_anglediff(best_turns, orientations))

			# rotation
			rotational_noise = torch.normal(mean=0., std=1., size=(amount,)) * torch.sqrt(2*self.DR)
			rotation = torch.exp((self.dt * self.Gamma * self.DR * torch.sin(angle_to_target) + rotational_noise*torch.sqrt(self.dt))*1.j)

			# translation
			translational_noise = torch.normal(mean=0., std=1., size=(amount,), dtype=torch.cfloat) * torch.sqrt(2*self.DT)
			translation = self.dt * self.velocity * orientations + translational_noise * torch.sqrt(self.dt)

		# apply changes
		positions = self.solve_collisions(positions + translation)
		orientations *= rotation

		return positions, orientations, orientation_sums, leftturns, rightturns


	def solve_collisions(self, positions, depth=0):
		"""Takes a complex vector of particle positions and
		simultaneously translates all overlapping particles away
		from eachother by the 1/n'th overlap distance,
		recursively until no more overlap or max 1000 times.
		"""
		alldists_compl = ActiveParticles.getdistances(positions, True)
		alldists_abs = alldists_compl.abs()
		all_collisions = alldists_abs <= 2*self.Rc - torch.eye(len(alldists_abs))
		overlap_distance = 2.1*self.Rc - alldists_abs
		# TODO why isnt this breaking for alldists_abs == 0?
		move_distance = torch.where(all_collisions, (alldists_compl/alldists_abs) * (overlap_distance/2), torch.tensor([0.+0.j]))
		positions -= move_distance.sum(dim=1)

		if all_collisions.sum() and depth < 1000:
			return self.solve_collisions(positions, depth + 1)
		else:
			return positions


	def O_R(self, positions, orientations):

		cm = positions.mean()
		r = positions-cm
		r /= r.abs()
		r = F.pad(torch.view_as_real(r), pad=(0,1,0,0), mode="constant", value=0)
		u = orientations / orientations.abs()
		u = F.pad(torch.view_as_real(u), pad=(0,1,0,0), mode="constant", value=0)
		e_z = torch.tensor([0,0,1], dtype=torch.float)

		return torch.mv(torch.cross(r, u), e_z)


	def local_O_R(self, measurement_positions, positions, orientations):

		dists = torch.cdist(torch.view_as_real(measurement_positions), torch.view_as_real(positions))
		exp_dists = torch.exp(-dists.abs() ** 2 / (2 * self.diameter ** 2))

		return torch.mv(exp_dists, self.O_R(positions, orientations)) / (exp_dists.sum(axis=1) + 1e-14)


	def O_P(self, orientations):

		return orientations.sum(axis=0).abs().sum() / len(orientations)
