"""
README:
These experiments expect a sub-directory called 'energy' where they save the results.
.gitignore prevents version control of the result files.
In the main code section, uncomment
    run_compute_fiber_energy_pool(args_list, num_procs=0)
to run the experiments and (re)generate the result files.
Once the result files exist, recomment this line if you only want to plot the results.
Edit args_list to modify the parameters of the experimental runs.

"""

import sys, time
import pickle as pk
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import cont_hopnet_customizable as chc
import rnn_fxpts as rfx

def make_random_hopnet(n, k, m, g):
    size = (n, k)
    data = (m + (1-m)*np.random.random(size))*np.random.choice([-1,1], size=size)
    hn = chc.Hopnet(n=n, gain=g)
    hn.learn(data)
    return hn, data

def plot_energy():

    n = 48 # network size
    k = 3 # number of training vectors
    m = .9 # lower bound on fixed point coordinate magnitude
    g = 8.0 # gain
    num_runs = 10 # number of trajectories to run
    num_steps = 10 # number of time-steps for each trajectory

    hn, data = make_random_hopnet(n, k, m, g)

    # Run network, tracking activity and energy
    activities = [[] for run in range(num_runs)]
    energies = [[] for run in range(num_runs)]
    for run in range(num_runs):

        # Random initial point near a corner
        hn.a = (m + (1-m)*np.random.random(n))*np.random.choice([-1,1], size=n)

        # Run dynamics
        for step in range(num_steps):
            activities[run].append(hn.a)
            energies[run].append(hn.energy(hn.a, None))
            hn.a = hn.update()

    # Show results
    fig = pt.figure(figsize=(10,6))
    act_ax = fig.add_subplot(1,2,1, projection="3d")
    eng_ax = fig.add_subplot(1,2,2)

    activities = np.array(activities)
    for run in range(num_runs):
        for step in range(num_steps-1):
            act_ax.plot(
                *activities[run,step:step+2,:3].T,
                linestyle='-',
                marker='.',
                color = .5*(1 - np.ones(4)*float(step)/num_steps))
    act_ax.set_xlabel('a_1')
    act_ax.set_xticks([-1,0,1])
    act_ax.set_ylabel('a_2')
    act_ax.set_yticks([-1,0,1])
    act_ax.set_zlabel('a_3')
    act_ax.set_zticks([-1,0,1])
    act_ax.set_title('Projected random trajectories')

    eng_ax.plot(np.array(energies).T, color='k')
    eng_ax.set_xlabel('Time step')
    eng_ax.set_ylabel('Energy')
    eng_ax.set_title('Energy along trajectories')

    pt.tight_layout()
    pt.show()

def compute_fiber_energy(
    filebase='tmp', # for saving results
    n = 48,         # network size
    k = 3,          # number of training vectors
    m = .9,         # lower bound on fixed point coordinate magnitude
    g = 8.0,        # gain
    steps = 1e6,    # max traverse steps
    ):

    hn, data = make_random_hopnet(n, k, m, g)

    # Run fiber solver
    logfile = open(filebase+'.log','w')
    fxpts, fiber = rfx.run_solver(g * hn.W, c=None, max_traverse_steps = steps, logfile = logfile)

    # Form and save results
    npz = {"g": g, "W": hn.W, "data":data, "fiber":fiber, "fxpts":fxpts}
    np.savez(filebase+'.npz', **npz)

    # Calculate stability, energy, spurious at fiber fixed points, and unfound data
    rfx.hardwrite(logfile,'Stability etc...\n')
    stable = np.zeros(fxpts.shape[1], dtype=bool)
    energy = np.empty(fxpts.shape[1])
    spurious = np.zeros(fxpts.shape[1], dtype=bool)
    found = np.zeros(data.shape[1], dtype=bool)
    for fp in range(fxpts.shape[1]):
        v = fxpts[:,fp]
        eigs = np.linalg.eigvals(hn.jacobian(v))
        stable[fp] = (np.absolute(eigs).max() < 1)

        energy[fp] = hn.energy(v, None)

        matches = (np.sign(v[:,np.newaxis] * data) > 0).all(axis=0)
        spurious[fp] = not matches.any()
        found |= matches

    # Update and save results
    npz["stable"] = stable
    npz["spurious"] = spurious
    npz["energy"] = energy
    npz["found"] = found
    np.savez(filebase+'.npz', **npz)

    logfile.close()

def compute_fiber_energy_caller(args): # one parameter for multiprocessing.Pool
    compute_fiber_energy(*args)

def run_compute_fiber_energy_pool(args_list, num_procs=0):

    if num_procs > 0: # multiprocessed

        num_procs = min(num_procs, mp.cpu_count())
        print("using %d processes..."%num_procs)
        pool = mp.Pool(processes=num_procs)
        pool.map(compute_fiber_energy_caller, args_list)
        pool.close()
        pool.join()

    else: # serial

        print("single processing...")
        for args in args_list: compute_fiber_energy_caller(args)

def plot_fiber_energy(filebase):

    npz = dict(np.load(filebase+'.npz'))
    g, W, data, fiber = float(npz["g"]), npz["W"], npz["data"], npz["fiber"]
    hn = chc.Hopnet(n=W.shape[0], gain=g)
    hn.W = W

    # Calculate energy along fiber
    energy = np.empty(fiber.shape[1])
    for step in range(fiber.shape[1]):
        energy[step] = hn.energy(fiber[:-1, step], None)

    # Plot energy and alpha along fiber
    pt.figure(figsize=(6,4))
    h1 = pt.plot(energy,'-k')[0]
    h2 = pt.plot(fiber[-1,:],'--k')[0]

    # Indicate zero-crossings
    sign_change = np.flatnonzero(np.sign(fiber[-1,:-1]*fiber[-1,1:]) <= 0)
    y_min = min(energy.min(), fiber[-1,:].min())
    y_max = max(energy.max(), fiber[-1,:].max())
    for sc in sign_change:
        pt.plot([sc, sc], [y_min, y_max], linestyle=':', color='gray', zorder=1)

    # Indicate stable and stored fixed points
    h3 = Line2D([0],[0], marker='s', c='none', markeredgecolor='k')
    h4 = Line2D([0],[0], marker='D', c='none', markerfacecolor='k', markeredgecolor='k')
    for sc in sign_change:
        v = fiber[:-1,sc]
        # Stability
        eigs = np.linalg.eigvals(hn.jacobian(v))
        if np.absolute(eigs).max() < 1: # stable around fixed point
            pt.scatter([sc, sc], [energy[sc], fiber[-1,sc]],
                marker='s', c='none', edgecolors='k', zorder=2)
        # Stored
        if (np.sign(data * v[:,np.newaxis]) > 0).all(axis=0).any():
            pt.scatter([sc, sc], [energy[sc], fiber[-1,sc]],
                marker='D', c='k', edgecolors='k', zorder=3)

    pt.xlabel('Traversal step')
    pt.ylabel('Energy and alpha')
    pt.legend([h1, h2, h3, h4], ['Energy','Alpha', 'Stable', 'Stored'])
    pt.show()

def plot_fxpt_energy(filebase):

    npz = dict(np.load(filebase+'.npz'))
    energy, stable, spurious = npz["energy"], npz["stable"], npz["spurious"]

    # Sort by energy
    idx = np.argsort(energy)
    energy = energy[idx]
    stable = stable[idx]
    spurious = spurious[idx]

    # Plot energies grouped by stable vs not, spurious vs not
    pt.figure(figsize=(5,4))
    h1 = pt.scatter(np.flatnonzero( stable & ~spurious),
        energy[ stable & ~spurious],
        marker='+', c='k', zorder=2)
    h2 = pt.scatter(np.flatnonzero( stable &  spurious),
        energy[ stable &  spurious],
        marker='o', c='none', edgecolors='k', zorder=2)
    h3 = pt.scatter(np.flatnonzero(~stable & ~spurious),
        energy[~stable & ~spurious],
        marker='+', c='gray', zorder=3)
    h4 = pt.scatter(np.flatnonzero(~stable &  spurious),
        energy[~stable &  spurious],
        marker='o', c='none', edgecolors='gray', zorder=1)

    pt.xticks(range(0, energy.size, int(np.ceil(energy.size/10.))))
    pt.legend([
        "Stable stored", "Stable spurious", "Unstable stored", "Unstable spurious"])
    pt.xlabel('Sorted Fixed points')
    pt.ylabel('Energy')
    pt.show()

def plot_energy_runs(filebases):

    pt.figure(figsize=(8,5))

    max_fxpts = 0
    for filebase in filebases:
        npz = dict(np.load(filebase+'.npz'))
        energy, stable, spurious = npz["energy"], npz["stable"], npz["spurious"]
        max_fxpts = max(max_fxpts, len(energy))

        # Sort by energy
        idx = np.argsort(energy)
        energy = energy[idx]
        stable = stable[idx]
        spurious = spurious[idx]

        # Connect all energies for current run
        pt.plot(energy, linestyle='-', color=(0.7,0.7,0.7), zorder=0)

        # Plot energies grouped by stable vs not, spurious vs not
        h1 = pt.scatter(np.flatnonzero( stable & ~spurious),
            energy[ stable & ~spurious],
            marker='+', c='k', zorder=2)
        h2 = pt.scatter(np.flatnonzero( stable &  spurious),
            energy[ stable &  spurious],
            marker='o', c='none', edgecolors='k', zorder=2)
        h3 = pt.scatter(np.flatnonzero(~stable & ~spurious),
            energy[~stable & ~spurious],
            marker='+', c='gray', zorder=3)
        h4 = pt.scatter(np.flatnonzero(~stable &  spurious),
            energy[~stable &  spurious],
            marker='o', c='none', edgecolors='gray', zorder=1)

    pt.xticks(range(0, max_fxpts, int(np.ceil(max_fxpts/10.))))
    pt.legend([h1, h2, h3, h4],
        ["Stable stored","Stable spurious","Unstable stored","Unstable spurious"])
    pt.xlabel('Sorted Fixed points')
    pt.ylabel('Energy')
    pt.show()

if __name__ == "__main__":

    plot_energy() # sanity check that energy decreases on one example network

    num_runs = 12
    args_list = [(
        'energy/run%d'%run,    # results filebase
        48,                    # network size
        3,                     # number of training vectors
        .9,                    # lower bound on fixed point coordinate magnitude
        8.0,                   # gain
        1e5,                   # max traverse steps
        ) for run in range(num_runs)]

    run_compute_fiber_energy_pool(args_list, num_procs=0)

    plot_fiber_energy(filebase='energy/run0')
    plot_fxpt_energy(filebase='energy/run0')
    plot_energy_runs(filebases=['energy/run%d'%r for r in range(num_runs)])

