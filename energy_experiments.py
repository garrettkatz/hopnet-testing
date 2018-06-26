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

def make_random_data(n, k, m):
    size = (n, k)
    data = (m + (1-m)*np.random.random(size))*np.random.choice([-1,1], size=size)
    return data

def make_hopnet(data, g, mode=None):
    if mode is None: mode = chc.modes["sync"]
    n = data.shape[0]
    hn = chc.Hopnet(n=n, gain=g, mode=mode)
    hn.learn(data)
    return hn

def plot_energy(mode_string):

    n = 100 # network size
    k = 3 # number of training vectors
    m = .9 # lower bound on fixed point coordinate magnitude
    g = 10.0 # gain
    mode = chc.modes[mode_string]
    
    num_runs = 10 # number of trajectories to run
    num_steps = 10 # number of time-steps for each trajectory

    data = make_random_data(n, k, m)
    hn = make_hopnet(data, g, mode=mode)

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
    filebase, # for saving results
    data,           # training data
    g = 10.0,        # gain
    mode=None,      # hopnet mode (sync or async)
    steps = 1e6,    # max traverse steps
    ):

    if mode is None: mode = chc.modes["sync"]
    hn = make_hopnet(data, g, mode=mode)

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
    filebase, n, k, m, g, max_traverse_steps = args
    # both modes
    data = make_random_data(n, k, m)
    for mode_string in ["sync", "async_deterministic"]:
        compute_fiber_energy(
            filebase + '_' + mode_string,
            data,
            g,
            chc.modes[mode_string],
            steps = max_traverse_steps,
        )

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
        for a,args in enumerate(args_list):
            print("%d of %d"%(a, len(args_list)))
            compute_fiber_energy_caller(args)

def plot_fiber_energy(filebase):

    npz = dict(np.load(filebase+'.npz'))
    g, W, data, fiber, found = float(npz["g"]), npz["W"], npz["data"], npz["fiber"], npz["found"]
    print("found %d of %d"%(np.count_nonzero(found), data.shape[1]))
    hn = chc.Hopnet(n=W.shape[0], gain=g)
    hn.W = W

    # Calculate energy along fiber
    energy = np.empty(fiber.shape[1])
    for step in range(fiber.shape[1]):
        energy[step] = hn.energy(fiber[:-1, step], None)

    # Plot energy along fiber
    pt.figure(figsize=(6,4))
    h1 = pt.plot(energy,'-k')[0]

    # Indicate zero-crossings and alpha mins
    sign_change = np.flatnonzero(np.sign(fiber[-1,:-1]*fiber[-1,1:]) <= 0)
    alpha_min = np.flatnonzero(
        (np.fabs(fiber[-1,:-2]) > np.fabs(fiber[-1,1:-1])) & \
        (np.fabs(fiber[-1,1:-1]) < np.fabs(fiber[-1,2:])))
    candidates = np.sort(list(set(sign_change) | set(alpha_min)))
    y_min, y_max = energy.min(), energy.max()
    # for sc in sign_change:
    for sc in candidates:
        pt.plot([sc, sc], [y_min, y_max], linestyle=':', color='gray', zorder=1)

    # Indicate stable and stored fixed points
    h2 = Line2D([0],[0], marker='s', c='none', markeredgecolor='k')
    h3 = Line2D([0],[0], marker='D', c='none', markerfacecolor='k', markeredgecolor='k')
    # for sc in sign_change:
    for sc in candidates:
        v = fiber[:-1,[sc]]
        # v, _ = rfx.post_process_fxpts(g*W, v)
        v = v[:,0]
        # Stability
        eigs = np.linalg.eigvals(hn.jacobian(v))
        if np.absolute(eigs).max() < 1: # stable around fixed point
            pt.scatter([sc], [energy[sc]],
                marker='s', c='none', edgecolors='k', zorder=2)
        # Stored
        if (np.sign(data * v[:,np.newaxis]) > 0).all(axis=0).any():
            pt.scatter([sc], [energy[sc]],
                marker='D', c='k', edgecolors='k', zorder=3)
        # Fiber solver also returns complements
        if (np.sign(data * v[:,np.newaxis]) < 0).all(axis=0).any():
            pt.scatter([sc], [energy[sc]],
                marker='D', c='k', edgecolors='k', zorder=3)

    pt.xlabel('Traversal step')
    pt.ylabel('Energy')
    pt.legend([h1, h2, h3], ['Energy', 'Stable', 'Stored'])
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

    # # # sanity check that energy decreases on one example network
    # plot_energy("sync")
    # plot_energy("async_deterministic")

    num_runs = 25
    args_list = [(
        'energy/run%d'%run,    # results filebase
        100,                    # network size
        3,                     # number of training vectors
        .9,                    # lower bound on fixed point coordinate magnitude
        10.0,                   # gain
        1e5,                   # max traverse steps
        ) for run in range(num_runs)]

    # run_compute_fiber_energy_pool(args_list, num_procs=0)

    for mode_string in ["sync", "async_deterministic"]:
        for d in range(num_runs):
            print("%s run %d"%(mode_string, d))
            plot_fiber_energy(filebase='energy/run%d_%s'%(d,mode_string))
            # plot_fxpt_energy(filebase='energy/run%d_%s'%(d,mode_string))
        # plot_energy_runs(filebases=['energy/run%d_%s'%(r, mode_string)
        #     for r in range(num_runs)])

