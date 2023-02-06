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

		# Equation settings
		self.init_func = options['initFunction']
		self.boundary_cond = options['boundaryCondition']

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
		self.q = None	# Current conservative variables
		self.q0 = None	# Initial conservative variables
		self.f = None	# Numerical diffusion

		try:
			if options['equationType'] == 'LinearScalarAdvectionEquation':
				self.c = options['fluxSpeed']
				self.hist = []	# To save solutions every iteration

			elif options['equationType'] == 'EulerEquation':
				self.gamma = options['gamma']
				self.ndmax = options['ndmax']
				self.qL = None # left cell interface
				self.qR = None # right cell interface
				self.var = None # Current primitive variables (r,u,p)
				self.var0 = None # Initial primitive variables (r,u,p)
				self.hist1 = []
				self.hist2 = []
				self.hist3 = []
				self.a = None # Speed of sound

			else:
				raise Exception("ERROR!!! The defined equation type is not available at the moment!")

		except KeyError:
			raise Exception("ERROR!!! Please define the equation type to be solved!")

	def save_sol(self, outputFolder):
		if self.options['equationType'] == 'LinearScalarAdvectionEquation':
			np.savetxt(outputFolder + "solution.dat", self.hist)
		elif self.options['equationType'] == 'EulerEquation':
			np.savetxt(outputFolder + "r_solution.dat", self.hist1)
			np.savetxt(outputFolder + "u_solution.dat", self.hist2)
			np.savetxt(outputFolder + "p_solution.dat", self.hist3)

class alfiSolver():
	"""
	Create an alfiSolver object
		options: dict

	"""
	def __init__(self, options):
		super(alfiSolver, self).__init__()
		self.options = options

	def initialize(self, problem):

		problem.jmax = self.options['jmax']
		problem.nmax = self.options['nmax']
		problem.x0 = self.options['x0']
		problem.x1 = self.options['x1']

		# Parameter setting and initialization
		problem.dx = (problem.x1 - problem.x0) / (problem.jmax - 1)

		# Generating x coordinates
		problem.x = np.linspace(problem.x0, problem.x1, problem.jmax)

		if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
			problem.dt = self.options['cfl'] * problem.dx / problem.c

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

		elif problem.options['equationType'] == 'EulerEquation':

			# Zero clear
			problem.q = np.zeros((problem.jmax,3))
			problem.q0 = np.zeros((problem.jmax,3))
			problem.var = np.zeros((problem.jmax,3))
			problem.var0 = np.zeros((problem.jmax,3))
			problem.s = np.zeros((problem.jmax,3))
			problem.f = np.zeros((problem.jmax,3))
			problem.qL = np.zeros((problem.jmax-1,3))
			problem.qR = np.zeros((problem.jmax-1,3))

			# Initial scalar q profile
			for j in range(problem.jmax):
				problem.q0[j] = problem.init_func(problem.x[j])

			problem.q = problem.q0

			# Initial speed of sound
			problem.var0[:,0] = problem.q0[:,0]
			problem.var0[:,1] = problem.q0[:,1]/problem.q0[:,0]
			problem.var0[:,2] = (problem.gamma-1)*(problem.q0[:,2]-0.5*problem.q0[:,0]*problem.var0[:,1]**2)
			problem.a = np.sqrt(problem.gamma*problem.var0[:,2]/problem.var0[:,0])
			# Initial dt
			# problem.dt = self.options['cfl']*problem.dx/max(abs(problem.var0[:,1])+problem.a)
			problem.dt = self.options['dt']

			# The problem is initialized
			problem.is_initialized = True

		else:
			pass

	def solve(self, problem, verbose=True):

		# Check if the problem has been initialized
		if problem.is_initialized:
			if self.options['saveHistory']:
				if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
					problem.hist.append([0.0] + list(problem.x))
					problem.hist.append([0.0] + list(problem.q0))
				elif problem.options['equationType'] == 'EulerEquation':
					problem.hist1.append([0.0] + list(problem.x))
					problem.hist2.append([0.0] + list(problem.x))
					problem.hist3.append([0.0] + list(problem.x))
					problem.hist1.append([0.0] + list(problem.var0[:,0]))
					problem.hist2.append([0.0] + list(problem.var0[:,1]))
					problem.hist3.append([0.0] + list(problem.var0[:,2]))

		else:
			raise Exception("ERROR!!! The problem has not been initialized yet.")

		if verbose:
			p1 = np.where(problem.x==0.00)[0][0]
			p2 = np.where(problem.x==0.25)[0][0]
			p3 = np.where(problem.x==0.50)[0][0]
			p4 = np.where(problem.x==0.75)[0][0]
			p5 = np.where(problem.x==1.00)[0][0]
			if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
				print("================================================================================")
				print(" Iter |  Time  | u (x=0.00) | u (x=0.25) | u (x=0.50) | u (x=0.75) | u (x=1.00) |")
				print("================================================================================")
				print(f"   0  | 0.0000 |   {problem.q0[p1] :.4f}   |   {problem.q0[p2] :.4f}   |   {problem.q0[p3] :.4f}   |   {problem.q0[p4] :.4f}   |   {problem.q0[p5] :.4f}   |")
			elif problem.options['equationType'] == 'EulerEquation':
				print("================================================================================")
				print(" Iter |  Time  | p (x=0.00) | p (x=0.25) | p (x=0.50) | p (x=0.75) | p (x=1.00) |")
				print("================================================================================")
				print(f"   0  | 0.0000 |   {problem.var0[p1,2] :.4f}   |   {problem.var0[p2,2] :.4f}   |   {problem.var0[p3,2] :.4f}   |   {problem.var0[p4,2] :.4f}   |   {problem.var0[p5,2] :.4f}   |")			

		# Core routines
		for n in range(problem.nmax):
			self._bc(problem)	# Boundary condition
			self._step(problem)	# Step operation

			if self.options['saveHistory']:
				if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
					problem.hist.append([problem.dt * (n+1)] + list(problem.q))
				elif problem.options['equationType'] == 'EulerEquation':
					problem.hist1.append([problem.dt * (n+1)] + list(problem.var[:,0]))
					problem.hist2.append([problem.dt * (n+1)] + list(problem.var[:,1]))
					problem.hist3.append([problem.dt * (n+1)] + list(problem.var[:,2]))

			if verbose:
				if n+1 < 10:
					if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
						print(f"   {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1] :.4f}   |   {problem.q[p2] :.4f}   |   {problem.q[p3] :.4f}   |   {problem.q[p4] :.4f}   |   {problem.q[p5] :.4f}   |")
					elif problem.options['equationType'] == 'EulerEquation':
						print(f"   {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1,2] :.4f}   |   {problem.q[p2,2] :.4f}   |   {problem.q[p3,2] :.4f}   |   {problem.q[p4,2] :.4f}   |   {problem.q[p5,2] :.4f}   |")
				elif n+1 < 100 or n==10:
					if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
						print(f"  {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1] :.4f}   |   {problem.q[p2] :.4f}   |   {problem.q[p3] :.4f}   |   {problem.q[p4] :.4f}   |   {problem.q[p5] :.4f}   |")
					elif problem.options['equationType'] == 'EulerEquation':
						print(f"  {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1,2] :.4f}   |   {problem.q[p2,2] :.4f}   |   {problem.q[p3,2] :.4f}   |   {problem.q[p4,2] :.4f}   |   {problem.q[p5,2] :.4f}   |")
				else:
					if problem.options['equationType'] == 'LinearScalarAdvectionEquation':
						print(f" {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1] :.4f}   |   {problem.q[p2] :.4f}   |   {problem.q[p3] :.4f}   |   {problem.q[p4] :.4f}   |   {problem.q[p5] :.4f}   |")
					elif problem.options['equationType'] == 'EulerEquation':
						print(f" {(n+1)}  | {problem.dt * (n+1) :.4f} |   {problem.q[p1,2] :.4f}   |   {problem.q[p2,2] :.4f}   |   {problem.q[p3,2] :.4f}   |   {problem.q[p4,2] :.4f}   |   {problem.q[p5,2] :.4f}   |")

			# Updated speed of sound
			if problem.options['equationType'] == 'EulerEquation':
				problem.var[:,0] = problem.q[:,0]
				problem.var[:,1] = problem.q[:,1]/problem.q[:,0]
				problem.var[:,2] = (problem.gamma-1)*(problem.q[:,2]-0.5*problem.var[:,0]*problem.var[:,1]**2)
				problem.a = np.sqrt(problem.gamma*problem.var[:,2]/problem.var[:,0])
				# Updated dt
				# problem.dt = self.options['cfl']*problem.dx/max(abs(problem.var[:,1])+problem.a)

		if verbose: print("================================================================================")

	def _bc(self, problem):
		problem.q = problem.boundary_cond(problem.q, problem.jmax)

	def _step(self, problem):

		try:
			if self.options['type'] == 'FiniteDifferenceMethod':
				self._deriv(problem) # Evaluating derivative
				# Spatial difference over control volume
				for j in range(1,problem.jmax-1):
					problem.s[j] = - problem.dt * problem.f[j]
			
			elif self.options['type'] == 'FiniteVolumeMethod':
				self._muscl(problem) # MUSCL + Limiter
				self._flux(problem)  # numerical flux
				
				# Spatial difference over control volume
				for j in range(1,problem.jmax-1):
					problem.s[j] = - problem.dt / problem.dx * (problem.f[j] - problem.f[j-1])

		except KeyError:
			raise Exception("ERROR!!! Please define the solver type to be used!")

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

	def _muscl(self, problem):

		# Compute and limit slopes
		dqL = np.zeros((problem.jmax-1,3))
		dqR = np.zeros((problem.jmax-1,3))

		for nd in range(problem.ndmax):
			for j in range(1, problem.jmax-2):
				
				# Left values
				num = problem.q[j,nd] - problem.q[j-1,nd]
				den = problem.q[j+1,nd] - problem.q[j,nd]
				if abs(num) < 1e-8:
					num=0.0; den=1.0;
				elif num > 1e-8 and abs(den) < 1e-8:
					num=1.0; den=1.0
				elif num <-1e-8 and abs(den) < 1e-8:
					num=-1.0; den=1.0
				dqL[j,nd] = psi(self.options['limiter'], num/den)

				# Right values
				num = problem.q[j+1,nd] - problem.q[j,nd]
				den = problem.q[j+2,nd] - problem.q[j+1,nd]
				if abs(num) < 1e-8:
					num=0.0; den=1.0;
				elif num > 1e-8 and abs(den) < 1e-8:
					num=1.0; den=1.0;
				elif num <-1e-8 and abs(den) < 1e-8:
					num=-1.0; den=1.0;
				dqR[j,nd] = psi(self.options['limiter'], num/den)

		# Left and Right extrapolated q-values at the boundary j+1/2
		for j in range(1,problem.jmax-2): # for the domain cells
			# 2nd-order piecewise linear
			problem.qL[j] = problem.q[j] + 0.5 * dqL[j] * (problem.q[j+1] - problem.q[j])
			problem.qR[j] = problem.q[j+1] - 0.5 * dqR[j] * (problem.q[j+2] - problem.q[j+1])

		# Boundary treatment (fixed to 1st order)
		problem.qL[0] = problem.q[0]
		problem.qR[0] = problem.q[0]
		problem.qL[problem.jmax-2] = problem.q[problem.jmax-2]
		problem.qR[problem.jmax-2] = problem.q[problem.jmax-2]

	def _flux(self, problem):
		gami = problem.gamma - 1.0

		for j in range(problem.jmax-1):
			### Left interface quantities ###
			r_L = problem.qL[j,0]
			u_L = problem.qL[j,1]/r_L
			e_L = problem.qL[j,2]
			p_L = gami*(e_L - 0.5*r_L*u_L**2)
			H_L = (problem.gamma/gami)*p_L/r_L + 0.5*u_L**2

			El = np.array([r_L * u_L,
						   r_L * u_L**2 + p_L,
						   r_L * u_L * H_L])

			Ql = np.array([r_L, r_L*u_L, e_L])

			### Right interface quantities ###
			r_R = problem.qR[j,0]
			u_R = problem.qR[j,1]/r_R
			e_R = problem.qR[j,2]
			p_R = gami*(e_R - 0.5*r_R*u_R**2)
			H_R = (problem.gamma/gami)*p_R/r_R + 0.5*u_R**2

			Er = np.array([r_R * u_R,
						   r_R * u_R**2 + p_R,
						   r_R * u_R * H_R])

			Qr = np.array([r_R, r_R*u_R, e_R])

			if p_L<0 or p_R<0:
				print(p_L, p_R)
				raise Exception('negative pressure found!')

			if r_L<0 or r_R<0:
				print(r_L, r_R)
				raise Exception('negative density found!')

			### Roe-averaged quantities ###
			u_tilde = (np.sqrt(r_R) * u_R + np.sqrt(r_L) * u_L) / (np.sqrt(r_R) + np.sqrt(r_L))
			H_tilde = (np.sqrt(r_R) * H_R + np.sqrt(r_L) * H_L) / (np.sqrt(r_R) + np.sqrt(r_L))
			c_tilde = np.sqrt(gami * (H_tilde - 0.5*u_tilde**2))

			### Eigenvalues ###
			eigenval_M = np.zeros((3,3))
			eigenval_M[0,0] = abs(u_tilde - c_tilde)
			eigenval_M[1,1] = abs(u_tilde)
			eigenval_M[2,2] = abs(u_tilde + c_tilde)

			### (QR - QL) ###
			Q_diff = Qr - Ql

			# Multiply with left eigenvector: R^{-1}(QR - QL)
			R_inv = np.array([[0.25*gami*u_tilde**2/c_tilde**2 + 0.5*u_tilde/c_tilde,
							   -0.5*gami*u_tilde/c_tilde**2 - 0.5/c_tilde,
							   0.5*gami/c_tilde**2],
							  [-0.5*gami*u_tilde**2/c_tilde**2 + 1,
							   gami*u_tilde/c_tilde**2,
							   -gami/c_tilde**2],
							  [0.25*gami*u_tilde**2/c_tilde**2 - 0.5*u_tilde/c_tilde,
							   -0.5*gami*u_tilde/c_tilde**2 + 0.5/c_tilde,
							   0.5*gami/c_tilde**2]])

			diss = np.matmul(R_inv,Q_diff)

			# Multiply with eigenvalues: AR^{-1}(QR - QL)
			diss = np.matmul(eigenval_M,diss)

			# Multiply with right eigenvector: RAR^{-1}(QR - QL)
			R = np.array([[1, 1, 1],
						  [u_tilde-c_tilde, u_tilde, u_tilde+c_tilde],
						  [0.5*u_tilde**2 + c_tilde**2/gami - u_tilde*c_tilde,
						   0.5*u_tilde**2,
						   0.5*u_tilde**2 + c_tilde**2/gami + u_tilde*c_tilde]])

			diss = np.matmul(R,diss)

			# Numerical flux
			problem.f[j] = 0.5*(Er + El - diss)
			

	def _lhs(self, problem):
		# Update the solutions
		for j in range(1, problem.jmax-1):
			problem.q[j] = problem.q[j] + problem.s[j]

def psi(limiter, r):
	"""
	Calculate psi(r) according to the limiter type
	"""
	if limiter == 'van_Albada1':
		psi_r = (r**2+r)/(r**2+1)

	elif limiter == 'van_Albada2':
		psi_r = 2*r/(r**2+1)

	elif limiter == 'minmod':
		psi_r = np.maximum(0, np.minimum(1, r))

	elif limiter == 'van_Leer':
		psi_r = (r+abs(r))/(1+abs(r))

	elif limiter == 'superbee':
		psi_r = np.max([0, np.minimum(2*r,1), np.minimum(r,2)])

	elif limiter == 'charm':
		if r <= 0: psi_r = 0.0
		else: psi_r = r*(3*r+1)/(r+1)**2

	elif limiter == 'ospre':
		psi_r = 1.5*(r**2+r)/(r**2+r+1) 

	elif limiter == 'hcus':
		psi_r = 1.5*(r+abs(r))/(r+2)

	elif limiter == 'smart':
		psi_r = np.maximum(0, np.min([2*r, 0.25+0.75*r, 4]))

	return psi_r

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

def plot_solution(folderName, equationType, time):
	"""
	Plot solutions at given time(s)
	If the given time is not available, find the closest data.

	Arguments:
		filename: name of the file containing solutions data
		time: list of time(s)
	"""
	if equationType == 'LinearScalarAdvectionEquation':
		data = np.genfromtxt(folderName + "solution.dat")
		for t in time:
			idx = np.abs(data[1:,0]-t).argmin() + 1
			plt.plot(data[0,1:],data[idx,1:],'o-',label=f't = {data[idx,0]}')

		plt.xlabel('x')
		plt.ylabel('u')
		plt.title("Solution for u(x,t)")
		plt.legend(loc="upper right")
		# plt.ylim([0,1])
		plt.show()
	elif equationType == 'EulerEquation':
		data_r = np.genfromtxt(folderName + "r_solution.dat")
		data_u = np.genfromtxt(folderName + "u_solution.dat")
		data_p = np.genfromtxt(folderName + "p_solution.dat")
		t = time[0]
		idx = np.abs(data_r[1:,0]-t).argmin() + 1
		plt.plot(data_r[0,1:],data_r[idx,1:],'b-',label=f'density')
		plt.plot(data_u[0,1:],data_u[idx,1:],'g-',label=f'velocity')
		plt.plot(data_p[0,1:],data_p[idx,1:],'r-',label=f'pressure')
		plt.xlabel('x')
		plt.title(f"Density, velocity, and pressure solution at t = {data_r[idx,0]}")
		plt.legend(loc='center left')
		plt.show()

def plot_comparison(folderName, equationType, limiter, var, time):
	if equationType == 'EulerEquation':
		data_r = []
		data_u = []
		data_p = []
		for _, name in enumerate(limiter):
			data_r.append(np.genfromtxt(folderName + name + '/r_solution.dat'))
			data_u.append(np.genfromtxt(folderName + name + '/u_solution.dat'))
			data_p.append(np.genfromtxt(folderName + name + '/p_solution.dat'))
		t = time[0]
		idx = np.abs(data_r[0][1:,0]-t).argmin() + 1
		for i, name in enumerate(limiter):
			plt.plot(data_r[i][0,1:],data_r[i][idx,1:],label=name)
			plt.title("Density")
			plt.legend(loc='upper right')
		plt.show()
		for i, name in enumerate(limiter):
			plt.plot(data_u[i][0,1:],data_u[i][idx,1:],label=name)
			plt.title("Velocity")
			plt.legend(loc='upper left')
		plt.show()
		for i, name in enumerate(limiter):
			plt.plot(data_p[i][0,1:],data_p[i][idx,1:],label=name)
			plt.title("Pressure")
			plt.legend(loc='upper right')
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

def plot_limiter_characteristics():
	r = np.linspace(0,3,100)
	minmod = psi('minmod', r)
	van_Albada1 = psi('van_Albada1', r)
	van_Albada2 = psi('van_Albada2', r)
	van_Leer = psi('van_Leer', r)
	superbee = np.zeros_like(r)
	for i in range(len(r)):
		superbee[i] = psi('superbee', r[i])
	plt.plot(r,minmod,label='minmod')
	plt.plot(r,van_Albada1,label='van_Albada1')
	plt.plot(r,van_Albada2,label='van_Albada2')
	plt.plot(r,van_Leer,label='van_Leer')
	plt.plot(r,superbee,label='superbee')
	plt.fill_between(r,minmod,superbee,color='lightgray',label='2nd-order TVD region')
	plt.xlabel('$r$')
	plt.ylabel(f'$\psi(r)$')
	plt.title('Limiter characteristics')
	plt.legend(loc='upper left')
	plt.show()