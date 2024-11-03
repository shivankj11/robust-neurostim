import numpy as np
import scipy

import pyomo
import pyomo.environ as pe
import romodel as ro
from romodel.uncset import EllipsoidalSet

import TransferFunction_chaitanya as tfc

def get_taufn():
    sigma = np.asarray([-np.inf, 1, 1/80, 1])
    radii = np.asarray([-np.inf, 10-1.1, 10-.6, 10])
    l = np.arange(300) + 1
    # ts = lambda ps : [calc_T(l, sigma * p, radii) for p in ps]
    # sample_ts = lambda r, ps : np.asarray([t(r) for t in ts(ps)])

    taufn = lambda u : tfc.TransferFunction(sigma[1:] + 
        sigma[1:] * np.random.normal(0, u, sigma[1:].shape), radii[1:])
    tau = lambda u : lambda r : taufn(u).calc_tauL(r = r)[1]

    return tau


m = pe.ConcreteModel()
# Create an indexed variable
m.x = pe.Var([0, 1])

# Create a generic uncertainty set
m.uncset = ro.UncSet()
# Define ellipsoidal set
# m.uncset = EllipsoidalSet(cov=[[1, 0, 0],
#                                [0, 1, 0],
#                                [0, 0, 1]],
#                           mean=[0.5, 0.3, 0.1])
# Create an indexed uncertain parameter
m.w = ro.UncParam([0, 1], uncset=m.uncset, nominal=[0.5, 0.8])
# Add constraints to the uncertainty set
m.uncset.cons1 = pe.Constraint(m.w[0] + m.w[1] <= 1.5)
m.uncset.cons2 = pe.Constraint(m.w[0] - m.w[1] <= 1.5)

# deterministic
m.x = pe.Var(range(3))
c = [0.1, 0.2, 0.3]
m.cons = pe.Constraint(expr=sum(c[i]*m.x[i] for i in m.x) <= 0)
# robust
m.x = pe.Var(range(3))
m.c = ro.UncParam(range(3), nominal=[0.1, 0.2, 0.3], uncset=m.uncset)
m.cons = pe.Constraint(expr=sum(m.c[i]*m.x[i] for i in m.x) <= 0)

# Define uncertain parameters
m.w = ro.UncParam(range(3), nominal=[1, 2, 3])
# Define adjustable variable which depends on uncertain parameter
m.y = ro.AdjustableVar(range(3), uncparams=[m.w], bounds=(0, 1))

# Set uncertain parameters for individual indicies
m.y[0].set_uncparams([m.w[0]])
m.y[1].set_uncparams([m.w[0], m.w[1]])

# Solve robust problem using reformulation
solver = pe.SolverFactory('romodel.reformulation')
solver.solve(m)
# Solve robust problem using cutting planes
solver = pe.SolverFactory('romodel.cuts')
solver.solve(m)
# Solve nominal problem
solver = pe.SolverFactory('romodel.nominal')
solver.solve(m)