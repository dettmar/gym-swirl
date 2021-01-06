import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
matplotlib.rcParams['animation.embed_limit'] = 2**32

class Animator:

	def __init__(self, env, start_frame=0, end_frame=int(1e4), lock_frame=False, show_arrows=True, scene_size=30):

		self.states = env.states
		self.particle_diam = env.aps.diameter # m
		self.scene_size = self.particle_diam*scene_size # m
		self.velocity = env.aps.velocity
		self.start_frame = start_frame
		self.end_frame = min(end_frame, len(self.states)-1)
		self.Ro = env.aps.Ro
		self.Ra = env.aps.Ra
		self.Rr = env.aps.Rr
		self.fig, self.ax = plt.subplots(figsize=(10,10))
		self.lock_frame = lock_frame
		self.show_arrows = show_arrows
		print(f"Start frame {self.start_frame} end frame: {self.end_frame}, total frames {self.end_frame-self.start_frame} states {len(self.states)}")
		self.anim = animation.FuncAnimation(self.fig, self.update,
											init_func=self.plot_init,
											interval=100, # ms
											frames=self.end_frame-self.start_frame)#,  blit=True)

	def plot_init(self):

		state = self.states[self.start_frame]
		decision_state = self.states[self.start_frame+1]
		median = state.positions.mean()
		center_x, center_y = median.real, median.imag
		self.ax.axis([center_x-self.scene_size, center_x+self.scene_size, center_y-self.scene_size, center_y+self.scene_size])
		M = self.ax.transData.get_matrix()
		xscale = M[0,0]
		yscale = M[1,1]

		# s= describes the area of the marker
		self.particles = self.ax.scatter(*torch.view_as_real(state.positions).T, s=(xscale*self.particle_diam)**2, color="black")#, 'o', markersize=10) #,
		if self.show_arrows:
			self.orientation_sum_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/20, length_includes_head=True, color="blue") for x, y, dx, dy in zip(*torch.view_as_real(decision_state.positions).T, *(torch.view_as_real(decision_state.orientation_sums/decision_state.orientation_sums.abs())).T*self.Ro)]
			self.left_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/20, length_includes_head=True, color="green") for x, y, dx, dy in zip(*torch.view_as_real(decision_state.positions).T, *torch.view_as_real(decision_state.leftturns/decision_state.leftturns.abs()).T*self.Ro)]
			self.right_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/20, length_includes_head=True, color="green") for x, y, dx, dy in zip(*torch.view_as_real(decision_state.positions).T, *torch.view_as_real(decision_state.rightturns/decision_state.rightturns.abs()).T*self.Ro)]
		self.orientation_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/10, length_includes_head=True, color="red") for x, y, dx, dy in zip(*torch.view_as_real(state.positions).T, *torch.view_as_real(state.orientations).T*self.particle_diam/2)]

		self.orientation_radius = self.ax.scatter(*torch.view_as_real(state.positions).T, s=(2*xscale*self.Ro)**2, facecolors='none', edgecolors='blue')
		self.attraction_radius = self.ax.scatter(*torch.view_as_real(state.positions).T, s=(2*xscale*self.Rr)**2, facecolors='none', edgecolors='gray')
		self.cm = self.ax.scatter(*torch.view_as_real(state.positions.mean()).T, s=(xscale*self.particle_diam/2)**2, color="green")
		if self.show_arrows:
			self.nrs = [self.ax.annotate(i, torch.view_as_real(pos), color="white") for i, pos in enumerate(state.positions)]

		return self.particles, self.orientation_radius, self.attraction_radius, self.cm#, self.annot,

	def complex2vec(self, comp, norm=False):
		if norm: comp /= np.absolute(comp)
		return np.array([comp.real, comp.imag])

	def remove(self, objs):
		for x in objs: x.remove()

	def update(self, i):

		state = self.states[i+self.start_frame]
		decision_state = self.states[i+self.start_frame+1]

		self.remove(self.orientation_arrs)
		if self.show_arrows:
			self.remove(self.left_arrs)
			self.remove(self.right_arrs)
			self.remove(self.orientation_sum_arrs)
			self.remove(self.nrs)

			self.nrs = [self.ax.annotate(i, torch.view_as_real(pos).T, color="white") for i, pos in enumerate(state.positions)]

			self.orientation_sum_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/20, length_includes_head=True, color="blue") for x, y, dx, dy in zip(*torch.view_as_real(decision_state.positions).T, *torch.view_as_real(decision_state.orientation_sums/decision_state.orientation_sums.abs()).T*self.Ro)]
			self.left_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/20, length_includes_head=True, color="green") for x, y, dx, dy in zip(*torch.view_as_real(decision_state.positions).T, *torch.view_as_real(decision_state.leftturns/decision_state.leftturns.abs()).T*self.Ro)]
			self.right_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/20, length_includes_head=True, color="green") for x, y, dx, dy in zip(*torch.view_as_real(decision_state.positions).T, *torch.view_as_real(decision_state.rightturns/decision_state.rightturns.abs()).T*self.Ro)]
		self.orientation_arrs = [self.ax.arrow(x, y, dx, dy, width=self.particle_diam/10, length_includes_head=True, color="red") for x, y, dx, dy in zip(*torch.view_as_real(state.positions).T, *torch.view_as_real(state.orientations).T*self.particle_diam/2)]

		self.particles.set_offsets(torch.view_as_real(state.positions))
		self.orientation_radius.set_offsets(torch.view_as_real(state.positions))
		self.attraction_radius.set_offsets(torch.view_as_real(state.positions))
		self.cm.set_offsets(torch.view_as_real(state.positions).mean(axis=0))

		self.ax.set_title(f"$O_R$ = {state.O_R.mean(dim=0):.2f}, $O_P$ = {state.O_P.sum():.2f}, T = {state.T:.1f}s, $\Delta$ = {int(state.Deltas.mean().item() * 180 / 3.1415926536)} $^\circ$, $\sigma_{{DT}}$ = {state.DT:.2e} $m^2 s^{{−1}}$ $\sigma_{{DR}}$ = {state.DR:.2e} $s^{{−1}}$")
		median = state.positions.mean()
		center_x, center_y = median.real, median.imag
		if not self.lock_frame:
			self.ax.set_xlim([center_x-self.scene_size, center_x+self.scene_size])
			self.ax.set_ylim([center_y-self.scene_size, center_y+self.scene_size])

		return self.particles, self.orientation_radius, self.attraction_radius, self.cm#, self.annot,

	def show(self):
		return HTML(self.anim.to_jshtml())

	def store(self, filename="anim"):
		self.anim.save(f"animations/{filename}.mp4", fps=10, extra_args=['-vcodec', 'libx264'])


if __name__ == "__main__":

	from activeparticles import APGroup

	aps = APGroup(100)
	aps.timesteps(1000, 100, 0.5)
	Animator(aps).store()
