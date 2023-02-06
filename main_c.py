# ==========================
# 		Import modules		
# ==========================

from alfiFlow import alfiSolver, alfiProblem
from alfiFlow import plot_solution, plot_comparison, plot_limiter_characteristics
import numpy as np

# ==============================
# 		Problem settings		
# ==============================

# Computational domain
x0 = 0.0 # lower limit
x1 = 1.0 # upper limit
jmax = 101 # number of points
tEnd = 0.2 # final time
dt = 0.0001 # time step
nmax = int(tEnd/dt) # max iter number
gam = 1.4 # specific heat ratio

# Initial solution
def init_func(x):
	f = np.zeros(3) # container
	if x < 0.5:
		f[0] = 1.0 			# rho
		f[1] = 0.0 			# rho*u
		f[2] = 1.0/(gam-1)  # e = p/(gamma-1) + 0.5*rho*u**2
	else:
		f[0] = 0.125 		# rho
		f[1] = 0.0   		# rho*u
		f[2] = 0.1/(gam-1)  # e = p/(gamma-1) + 0.5*rho*u**2
	return f

# Boundary condition
def boundary_cond(q, jmax):
	q[0,0] = 1.0 				# rho
	q[0,1] = 0.0 				# rho*u
	q[0,2] = 1.0/(gam-1)		# e
	q[jmax-1,0] = 0.125 		# rho
	q[jmax-1,1] = 0.0 			# rho*u
	q[jmax-1,2] = 0.1/(gam-1) 	# e
	return q

# ==========================================================
# 		Problem and Solver Description using alfiFlow		
# ==========================================================

problemOptions = {
	# Equation
	'equationType': 'EulerEquation',
	'initFunction': init_func,
	'boundaryCondition': boundary_cond,
	'gamma': gam,
	'ndmax': 3 # solve for 3 variables (d, v, p)
}

problem = alfiProblem(options=problemOptions)

solverOptions = {
	# Common
	'type': 'FiniteVolumeMethod',
	'scheme': 'ROE',
	'limiter': 'minmod',
	# 'limiter': 'van_Albada1',
	# 'limiter': 'van_Albada2',
	# 'limiter': 'van_Leer',
	# 'limiter': 'superbee',
	# 'limiter': 'charm',
	# 'limiter': 'ospre',
	# 'limiter': 'hcus',
	# 'limiter': 'smart',
	'saveHistory': True,

	# Grid and discretization
	'x0': x0,
	'x1': x1,
	'jmax': jmax,
	'nmax': nmax,
	'dt': dt
}

solver = alfiSolver(options=solverOptions)

# ==================================
# 		Solving the equations		
# ==================================

solver.initialize(problem)
solver.solve(problem, verbose=True)
problem.save_sol(filename="solution.dat")

# ======================================
# 		Visualizing the solutions		
# ======================================

plot_solution(filename="solution.dat",
			  equationType='EulerEquation',
			  time=[0.20])

limiter_list = ['minmod', 'van_Albada1', 'van_Albada2', 'van_Leer', 'superbee']

plot_comparison(filename="solution.dat",
				limiter=limiter_list,
			    equationType='EulerEquation',
			    var=["r","u","p"],
			    time=[0.20])

plot_limiter_characteristics()