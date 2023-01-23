# ==========================
# 		Import modules		
# ==========================

from alfiFlow import alfiSolver, alfiProblem, plot_solution
import numpy as np

# ==============================
# 		Problem settings		
# ==============================

# Computational domain
x0 = 0.0
x1 = 1.0
jmax = 101

# Wave Speed
c = 1.0

# CFL number: c * delta_t / delta_x
cfl = 0.5

# max iter number to evaluate at t = 0.5
nmax = int(0.5 / (cfl * ((x1-x0)/(jmax-1)) / c))

# Initial solution
def init_func(x):
	if x >= 0.05 and x <= 0.45:
		f = 0.5 * (1.0+np.cos(5*np.pi*(x - 0.25)))
	else:
		f = 0.0
	return f

# Boundary condition
def boundary_cond(q, jmax):
	q[0] = 0.0
	q[jmax-1] = 0.0
	return q

# ==========================================================
# 		Problem and Solver Description using alfiFlow		
# ==========================================================

problemOptions = {
	# Equation
	'equationType': 'LinearScalarAdvectionEquation',
	'fluxSpeed': c,
	'initFunction': init_func,
	'boundaryCondition': boundary_cond,
}

problem = alfiProblem(options=problemOptions)

solverOptions = {
	# Common
	# 'scheme': 'CentralDifference',
	# 'scheme': 'FirstOrderUpwindDifference',
	'scheme': 'LaxScheme',
	# 'scheme': 'LaxWendroffScheme',
	# 'scheme': 'SecondOrderUpwindDifference',
	# 'scheme': 'SecondOrderUpwindDifferenceModified',
	'saveHistory': True,

	# Grid and discretization
	'x0': x0,
	'x1': x1,
	'jmax': jmax,
	'nmax': nmax,

	# Stability
	'cfl': cfl
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

plot_solution(filename="solution.dat", time=[0.00, 0.50])