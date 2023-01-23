import numpy as np
import matplotlib.pyplot as plt

class alfiProblem():
	"""
	Create an alfiProblem object
		options: dict

	"""
	def __init__(self, options):
		super(alfiProblem, self).__init__()
		
		self.options = options
		self.is_initialized = False

		try:
			if options['equationType'] == 'LinearScalarAdvectionEquation':
				
				# Equation settings
				self.init_func = options['initFunction']
				self.boundary_cond = options['boundaryCondition']
				self.c = options['fluxSpeed']
				
				# Coordinates and discretization
				self.x = None
				self.nmax = None
				self.jmax = None
				self.x0 = None
				self.x1 = None
				self.dx = None
				self.dt = None
				self.s = None	# delta_u = u_(n+1) - u_(n)

				# Solutions
				self.q = None	# Current
				self.q0 = None	# Initial
				self.f = None	# Numerical diffusion
				self.hist = []	# To save solutions every iteration

			else:
				raise Exception("ERROR!!! The defined equation type is not available at the moment!")

		except KeyError:
			raise Exception("ERROR!!! Please define the equation type to be solved!")

	def save_sol(self, filename):
		np.savetxt(filename, self.hist)

class alfiSolver():
	"""
	Create an alfiSolver object
		options: dict

	"""
	def __init__(self, options):
		super(alfiSolver, self).__init__()
		self.options = options

	def initialize(self, problem):

		if problem.options['equationType'] == 'LinearScalarAdvectionEquation':

			problem.jmax = self.options['jmax']
			problem.nmax = self.options['nmax']
			problem.x0 = self.options['x0']
			problem.x1 = self.options['x1']

			# Parameter setting and initialization
			problem.dx = (problem.x1 - problem.x0) / (problem.jmax - 1)
			problem.dt = self.options['cfl'] * problem.dx / problem.c

			# Generating x coordinates
			problem.x = np.linspace(problem.x0, problem.x1, problem.jmax)

			# Zero clear
			problem.q = np.zeros(problem.jmax)
			problem.q0 = np.zeros(problem.jmax)
			problem.s = np.zeros(problem.jmax)
			problem.f = np.zeros(problem.jmax)

			# Initial scalar u profile
			for j in range(problem.jmax):
				problem.q0[j] = problem.init_func(problem.x[j])

			problem.q = problem.q0

			# The problem is initialized
			problem.is_initialized = True

		else:
			pass

	def solve(self, problem, verbose=True):

		# Check if the problem has been initialized
		if problem.is_initialized:
			if self.options['saveHistory']:
				problem.hist.append([0.0] + list(problem.x))
				problem.hist.append([0.0] + list(problem.q0))
		else:
			raise Exception("ERROR!!! The problem has not been initialized yet.")

		if verbose:
			p1 = np.where(problem.x==0.00)[0][0]
			p2 = np.where(problem.x==0.25)[0][0]
			p3 = np.where(problem.x==0.50)[0][0]
			p4 = np.where(problem.x==0.75)[0][0]
			p5 = np.where(problem.x==1.00)[0][0]
			print("================================================================================")
			print(" Iter |  Time  | u (x=0.00) | u (x=0.25) | u (x=0.50) | u (x=0.75) | u (x=1.00) |")
			print("================================================================================")
			print(f"   0  | 0.0000 |   {problem.q0[p1] :.4f}   |   {problem.q0[p2] :.4f}   |   {problem.q0[p3] :.4f}   |   {problem.q0[p4] :.4f}   |   {problem.q0[p5] :.4f}   |")

		# Core routines
		for n in range(problem.nmax):
			self._bc(problem)	# Boundary condition
			self._step(problem)	# Step operation

			if self.options['saveHistory']:
				problem.hist.append([problem.dt * (n+1)] + list(problem.q))

			if verbose:
				if n+1 < 10:
					print(f"   {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1] :.4f}   |   {problem.q[p2] :.4f}   |   {problem.q[p3] :.4f}   |   {problem.q[p4] :.4f}   |   {problem.q[p5] :.4f}   |")
				elif n+1 < 100 or n==10:
					print(f"  {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1] :.4f}   |   {problem.q[p2] :.4f}   |   {problem.q[p3] :.4f}   |   {problem.q[p4] :.4f}   |   {problem.q[p5] :.4f}   |")
				else:
					print(f" {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1] :.4f}   |   {problem.q[p2] :.4f}   |   {problem.q[p3] :.4f}   |   {problem.q[p4] :.4f}   |   {problem.q[p5] :.4f}   |")

		if verbose: print("================================================================================")

	def _bc(self, problem):
		problem.q = problem.boundary_cond(problem.q, problem.jmax)

	def _step(self, problem):
		# Evaluating derivative according to the scheme
		self._deriv(problem)

		# Spatial difference over control volume
		for j in range(1,problem.jmax-1):
			problem.s[j] = - problem.dt * problem.f[j]

		# Time advancement
		self._lhs(problem)

	def _deriv(self, problem):
		try:
			if self.options['scheme'] == 'CentralDifference':
				for j in range(problem.jmax-1):
					problem.f[j] = 0.5 * problem.c * (problem.q[j+1] - problem.q[j-1]) / problem.dx

			elif self.options['scheme'] == 'FirstOrderUpwindDifference':
				for j in range(problem.jmax-1):
					problem.f[j] = problem.c * (problem.q[j] - problem.q[j-1]) / problem.dx

			elif self.options['scheme'] == 'LaxScheme':
				v = problem.c * problem.dt / problem.dx
				for j in range(problem.jmax-1):
					problem.f[j] = 0.5 * ((v-1) * problem.q[j+1] - (v+1)*problem.q[j-1] + 2*problem.q[j]) / (problem.dt)

			elif self.options['scheme'] == 'LaxWendroffScheme':
				v = problem.c * problem.dt / problem.dx
				for j in range(problem.jmax-1):
					problem.f[j] = 0.5 * problem.c * ((1-v)*problem.q[j+1] - (1+v)*problem.q[j-1] + 2*v*problem.q[j]) / problem.dx 

			elif self.options['scheme'] == 'SecondOrderUpwindDifference':
				for j in range(problem.jmax-1):
					problem.f[j] = 0.5 * problem.c * (3*problem.q[j] - 4*problem.q[j-1] - problem.q[j-2]) / problem.dx

			elif self.options['scheme'] == 'SecondOrderUpwindDifferenceModified':
				for j in range(problem.jmax-1):
					first_term = problem.c * (problem.q[j] - problem.q[j-1]) / problem.dx
					second_term = 0.5 * (problem.c/problem.dx) * (1.0 - problem.c*problem.dt/problem.dx) * (problem.q[j] - 2*problem.q[j-1] + problem.q[j-2])
					problem.f[j] = first_term + second_term

			else:
				raise Exception("ERROR!!! The defined scheme is not available at the moment!")

		except KeyError:
			raise Exception("ERROR!!! Please define the scheme for the solver!")

	def _lhs(self, problem):
		# Update the solutions
		for j in range(1, problem.jmax-1):
			problem.q[j] = problem.q[j] + problem.s[j]

class vonNeumannStabilityAnalysis():
	"""
	Create a vonNeumannStabilityAnalysis object
	"""
	def __init__(self, jmax):
		super(vonNeumannStabilityAnalysis, self).__init__()
		self.jmax = int(jmax)
		self.theta = np.linspace(0.0, np.pi, jmax)
		self.x = None
		self.y = None
		self.cfl = None

		self.g_ref = None # amplitude factor exact
		self.p_ref = None # phase exact
		self.p_ref_rel = None # phase relative p_ref_rel = p_ref/p_ref

		self.g = None # amplitude factor
		self.p = None # phase
		self.p_rel = None # phase relative p_rel = p/p_ref

		# Unit circle for reference
		self.xc = np.cos(self.theta)
		self.yc = np.sin(self.theta)

	def do(self, scheme, cfl):

		self.cfl = cfl

		# Exact amplitude and phase
		self.g_ref = np.ones(self.jmax)
		self.p_ref = - cfl * self.theta
		self.p_ref_rel = np.ones(self.jmax)

		# Amplitude and phase calculation according to the scheme

		if scheme == "CentralDifference":
			self.g = np.sqrt(np.ones(self.jmax) + cfl * cfl * np.sin(self.theta) * np.sin(self.theta))
			self.p = - np.arctan(cfl * np.sin(self.theta))
			self.p_rel = self.p/self.p_ref
			self.p_rel[np.isnan(self.p_rel)] = self.p_rel[1] # forcing non-nan solution

		elif scheme == "FirstOrderUpwindDifference":
			self.g = np.sqrt(np.ones(self.jmax) + 2*cfl*(cfl-1.0)*(np.ones(self.jmax) - np.cos(self.theta)))
			self.p = - np.arctan((cfl * np.sin(self.theta))/(np.ones(self.jmax) - cfl + cfl*np.cos(self.theta)))
			self.p[np.where(self.p>0.0)] += -np.pi # shifting solutions for negative region
			self.p_rel = self.p/self.p_ref
			self.p_rel[np.isnan(self.p_rel)] = self.p_rel[1] # forcing non-nan solution

		elif scheme == "LaxScheme":
			self.g = np.sqrt(np.ones(self.jmax) - cfl * cfl * np.sin(self.theta) * np.sin(self.theta))
			self.p = - np.arctan(cfl * np.tan(self.theta))
			self.p[np.where(self.p>0.0)] += -np.pi # shifting solutions for negative region
			self.p_rel = self.p/self.p_ref
			self.p_rel[np.isnan(self.p_rel)] = self.p_rel[1] # forcing non-nan solution

		elif scheme == "LaxWendroffScheme":
			self.g = np.sqrt(np.square((np.ones(self.jmax)-cfl*cfl)+cfl*cfl*np.cos(self.theta)) + cfl*cfl*np.square(np.sin(self.theta)))
			self.p = - np.arctan((cfl * np.sin(self.theta))/((np.ones(self.jmax)-cfl*cfl)+np.cos(self.theta)))
			self.p[np.where(self.p>0.0)] += -np.pi # shifting solutions for negative region
			self.p_rel = self.p/self.p_ref
			self.p_rel[np.isnan(self.p_rel)] = self.p_rel[1] # forcing non-nan solution

		elif scheme == 'SecondOrderUpwindDifference':
			real = np.ones(self.jmax) - 0.5*cfl*(3*np.ones(self.jmax)-4*np.cos(self.theta)-np.cos(2*self.theta))
			imag = 0.5*cfl*(4*np.sin(self.theta)+np.sin(2*self.theta))
			self.g = np.sqrt(real*real + imag*imag)
			self.p = - np.arctan(imag/real)
			self.p[np.where(self.p>0.0)] += -np.pi # shifting solutions for negative region
			self.p_rel = self.p/self.p_ref
			self.p_rel[np.isnan(self.p_rel)] = self.p_rel[1] # forcing non-nan solution

		elif scheme == 'SecondOrderUpwindDifferenceModified':
			real = np.ones(self.jmax) - 1.5*cfl * 0.5*cfl*cfl + (2*cfl - cfl*cfl)*np.cos(self.theta) + 0.5*cfl*(cfl-1.0)*np.cos(2*self.theta)
			imag = (2*cfl - cfl*cfl)*np.sin(self.theta) + 0.5*cfl*(cfl-1.0)*np.sin(2*self.theta)
			self.g = np.sqrt(real*real + imag*imag)
			self.p = - np.arctan(imag/real)
			self.p[np.where(self.p>0.0)] += -np.pi # shifting solutions for negative region
			self.p_rel = self.p/self.p_ref
			self.p_rel[np.isnan(self.p_rel)] = self.p_rel[1] # forcing non-nan solution

		else:
			raise Exception("The defined scheme is currently unavailable!")

def plot_solution(filename, time):
	"""
	Plot solutions at given time(s)
	If the given time is not available, find the closest data.

	Arguments:
		filename: name of the file containing solutions data
		time: list of time(s)
	"""
	data = np.genfromtxt(filename)
	for t in time:
		idx = np.abs(data[1:,0]-t).argmin() + 1
		plt.plot(data[0,1:],data[idx,1:],'o-',label=f't = {data[idx,0]}')

	plt.xlabel('x')
	plt.ylabel('u')
	plt.title("Solution for u(x,t)")
	plt.legend(loc="upper right")
	# plt.ylim([0,1])
	plt.show()

def plot_stability(analyses, type):
	"""
	This plot function provides two types of stability graphs
	analyses: a list of von Neumann analysis objects
	type: "amplification", "phase", "complex"
	"""
	if type == "amplification":

		# Exact
		plt.polar(analyses[0].theta, analyses[0].g_ref, "ko",fillstyle="none",label="unit circle")
		
		# Amplitude factors
		for analysis in analyses:
			plt.polar(analysis.theta, analysis.g,'-', label=f"cfl = {analysis.cfl}")

		plt.xlim([0,np.pi])
		plt.legend(loc="upper right")
		plt.show()

	elif type == "phase":

		# Exact
		plt.polar(analyses[0].theta, analyses[0].p_ref_rel, "ko",fillstyle="none",label="unit circle")

		# Relative phases
		for analysis in analyses:
			plt.polar(analysis.theta, analysis.p_rel,'-', label=f"cfl = {analysis.cfl}")
		
		plt.xlim([0.0,np.pi])
		plt.legend(loc="upper right")
		plt.show()

	else:
		raise Exception("ERROR!!! The defined plot type is not available!!!")