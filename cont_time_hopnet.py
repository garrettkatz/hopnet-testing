""" 
Implement functions used by the new solver (directional-fibers)
to solve continuous-time continuous-valued Hopnet.
"""

import sys
import numpy as np
import dfibers.numerical_utilities as nu
import dfibers.traversal as tv
import dfibers.logging_utilities as lu


def get_solver_kwargs(data, c=None, step_size=None, ef=None, history=1000, silent=True):
    W = np.matmul(data, data.T)
    for i in range(W.shape[0]):
        W[i,i] = 0
    W *= (1.0/W.shape[0])

    logfile = sys.stdout

    fiber_kwargs = {
        "f": lambda v: _f(W,v),
        "ef": ef_factory(W) if ef is None else lambda v: ef*np.ones((W.shape[0],1)), # 1e-6 good default for ef
        "Df": lambda v: _Df(W,v),
        "compute_step_amount": compute_step_amount_factory(W) if step_size is None else lambda trace: (step_size, 0, 0),
        "v": None, # Just use default - origin
        "c": c,
        "N": data.shape[0],
        "terminate": None, # lambda trace: (np.fabs(trace.x[:N,:]) > 3).any(),
        "max_traverse_steps": 2e5 if step_size is None else int(200/step_size), # step size computed is normally in 1e-4 range and 200/1e-4 = 2e6. 2e5 is a compromise
        "max_solve_iterations": 2**5,
        "max_history": history,
        "logger": lu.Logger(logfile) if not silent else None
    }

    # Old kwargs:
    #     fiber_kwargs = {
    #         "f": lambda v: _f(W,v),
    #         "ef": lambda v: 1e-6*np.ones((W.shape[0],1)),
    #         "Df": lambda v: _Df(W,v),
    #         "compute_step_amount": lambda trace: (step_size, 0),
    #         "v": None, # Just use default - origin
    #         "c": None, # Just use default - random
    #         "N": data.shape[0],
    #         "terminate": None,
    #         #"max_step_size": 1,
    #         "max_traverse_steps": int(200/step_size),
    #         "max_solve_iterations": (2**5),
    #         "max_history": 1000,
    #         #"solve_tolerance": 10**-10,
    #         "logger": lu.Logger(logfile) if not silent else None
    #     }
    # 


    return fiber_kwargs

    



def _f(W,v):
    return np.dot(W,v)*(1.0-v**2)

def _Df(W,v):
    Sig = np.array([np.diag(1.0-v[:,i]**2) for i in xrange(v.shape[1])])
    Tau = np.array([np.diag(v[:,i]*np.dot(W,v[:,i])) for i in xrange(v.shape[1])])
    return np.matmul(Sig,W) - 2.0*Tau


def ef_factory(W):
    # return lambda V: 10**-6 * np.ones(V.shape)
    def ef(V):
        e1V2 = nu.eps(1-V**2) + \
            2*np.fabs(V)*nu.eps(V) + nu.eps(V**2) # error in 1-V**2
        eWV = V.shape[0]*nu.eps(np.fabs(W).dot(np.fabs(V))) + \
            np.fabs(W).dot(nu.eps(V)) # error in W.dot(V)        
        return nu.eps(W.dot(V)*(1-V**2)) + \
            np.fabs(W.dot(V))*e1V2 + \
            np.fabs(1-V**2)*eWV + \
            e1V2*eWV             
    return ef

def compute_step_amount_factory(W):

    # bound on |d**2f_i(v) / dv_j dv_k| for all i, j, k
    def f2(v):
        return 4*np.fabs(v).max()*np.fabs(W).max() + 2*np.fabs(W.dot(v)).max()

    # bound on |d**3f_i(v) / dv_j dv_k dv_l| for all v, i, j, k
    f3 = 6*np.fabs(W).max()

    # certified step size computation
    return tv.compute_step_amount_factory(f2, f3)