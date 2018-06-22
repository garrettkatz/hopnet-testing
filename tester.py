"""
WARNING: this file is a mess, don't touch it
"""

import sys
import os 
# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, dir_path+'/../rnn-fxpts')

import numpy as np
# import sympy as sp
import scipy.integrate as spi
import scipy.linalg as spl
# import async_cont_hopnet as ach
# import sync_cont_hopnet as sch
import cont_hopnet_customizable as chc
import cont_time_hopnet as cth
import gen_data as gd
import utils
# import symbolic_critical_point_finder as scpf
import dfibers.traversal as df_tv
import dfibers.numerical_utilities as df_nu
import dfibers.fixed_points as df_fx
import dfibers.solvers as df_sv
import rnn_fxpts as rf
import rnn_fxpts_thresh_pp as rftpp
# import rnn_fxpts_limited as rfl

import itertools as it
import time
import multiprocessing as mp

import traceback

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_zeroing():
    D=20
    dn = open(os.devnull, 'w')
    # hn = ach.Hopnet(D, gain=5)
    hn = sch.Hopnet(D, gain=5)
    nz_nc = 0
    nz_cz = 0
    z_nc = 0
    z_cz = 0

    for i in range(10000):
        data = np.random.random((D,3))*2-1
        W = np.zeros((D,D), dtype=np.float32)
        for j in range(data.shape[1]):
            W += np.outer(data[:,j],data[:,j])
        W *= 1.0/D
        hn.W=W
        for j in range(data.shape[1]):
            t,_ = hn.simhop(data[:,j], fileid=dn, trace=False)
            if t>=500:
                nz_nc += 1
            if np.allclose(hn.a, np.zeros(hn.a.shape), rtol=0, atol=1e-5):
                nz_cz += 1
        hn.learn(data)
        for j in range(data.shape[1]):
            t,_ = hn.simhop(data[:,j], fileid=dn, trace=False)
            if t>=500:
                z_nc += 1
            if np.allclose(hn.a, np.zeros(hn.a.shape), rtol=0, atol=1e-5):
                z_cz += 1

    print('Non-zeroed non-convergent: {}'.format(nz_nc))
    print('Non-zeroed converges to zero: {}'.format(nz_cz))
    print('Zeroed non-convergent: {}'.format(z_nc))
    print('Zeroed converges to zero: {}'.format(z_cz))


def test_gain_eval_traversal(D=30, N=4, data=None, vals=[0.5+i*0.5 for i in range(10)]):
    # D=30
    # hn = sch.Hopnet(D)
    if data is None:
        data = np.random.random((D,N))*2-1
    else:
        D,N = data.shape
    W = np.zeros((D,D), dtype=np.float32)
    for i in range(N):
        W += np.outer(data[:,i],data[:,i])
    for i in range(D):
        W[i,i] = 0

    num_fps = []
    fxpts_lst = []

    for v in vals:
        gain = v/np.max(np.linalg.eigvalsh(W))
        fxpts, fiber = rfl.run_solver(W*gain)
        num_fps.append((v, gain, fxpts.shape[1]))
        fxpts_lst.append(fxpts)

    print(num_fps)

    return num_fps, fxpts_lst


def test_gain_eval_baseline(D=30, N=4, data=None, vals=[0.5+i*0.5 for i in range(10)]):
    # D=30
    # hn = sch.Hopnet(D)
    if data is None:
        data = np.random.random((D,N))*2-1
    else:
        D,N = data.shape
    W = np.zeros((D,D), dtype=np.float32)
    for i in range(N):
        W += np.outer(data[:,i],data[:,i])
    for i in range(D):
        W[i,i] = 0

    num_fps = []
    fxpts_lst = []

    for v in vals:
        gain = v/np.max(np.linalg.eigvalsh(W))
        fxV, _ = rfl.baseline_solver(W*gain)
        fxV_unique, _ = rfl.post_process_fxpts(W*gain, fxV)
        num_fps.append((v, gain, fxV_unique.shape[1]))
        fxpts_lst.append(fxV_unique)

    print(num_fps)

    return num_fps, fxpts_lst

def test_dirs(gain, W):
    res = 0
    for v in it.product([-1,1], repeat=len(W)):
        vect = np.array(v)
        vect = vect/np.linalg.norm(vect)
        if np.linalg.norm(gain*np.dot(W,vect)) >= 1:
            res += 1
    return res




def test_properties(n, min_pos_e, orth, zero, scaled):
    if orth and scaled:
        raise ValueError('Cannot use both orthonormal and scaled data')

    min_evals_W = []
    min_evals_W_I = []
    num_pos_evals = []
    evect_hdists = []
    evect_resids = []
    num_fps = []
    fp_hdists = []
    # fp_lstsq = []
    fp_resids = []


    for i in range(n):
        if scaled:
            d1 = (np.random.random((20,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(20,3))
            d2 = (np.random.random((50,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(50,3))
            d3 = (np.random.random((50,10))/2.0 + 0.5)*np.random.choice([-1,1], size=(50,10))
        else:
            d1 = np.random.random((20,3))*2-1
            d2 = np.random.random((50,3))*2-1
            d3 = np.random.random((50,10))*2-1
        
            if orth:
                d1 = gd.gs(d1)
                d2 = gd.gs(d2)
                d3 = gd.gs(d3)

        W1 = np.matmul(d1,d1.T)
        W2 = np.matmul(d2,d2.T)
        W3 = np.matmul(d3,d3.T)

        if zero:
            for j in range(W1.shape[0]):
                W1[j,j] = 0
            for j in range(W2.shape[0]):
                W2[j,j] = 0
            for j in range(W3.shape[0]):
                W3[j,j] = 0

        W1_evals, W1_evects = np.linalg.eigh(W1)
        W2_evals, W2_evects = np.linalg.eigh(W2)
        W3_evals, W3_evects = np.linalg.eigh(W3)

        min_evals_W.append([np.min(np.abs(W1_evals)),
                            np.min(np.abs(W2_evals)),
                            np.min(np.abs(W3_evals))])
        min_evals_W_I.append([np.min(np.abs(np.linalg.eigvalsh(W1-np.eye(W1.shape[0])))),
                              np.min(np.abs(np.linalg.eigvalsh(W2-np.eye(W2.shape[0])))),
                              np.min(np.abs(np.linalg.eigvalsh(W3-np.eye(W3.shape[0]))))])
        num_pos_evals.append([np.sum(W1_evals>1e-7),
                              np.sum(W2_evals>1e-7),
                              np.sum(W3_evals>1e-7)])

        hdists1 = []
        hdists2 = []
        hdists3 = []
        for j in range(d1.shape[1]):
            min_hdist = 100
            for k in range(d1.shape[1]):
                hd = utils.hdist(d1[:,j], W1_evects[:,-1-k])
                if hd < min_hdist:
                    min_hdist = hd
            hdists1.append(min_hdist)
        for j in range(d2.shape[1]):
            min_hdist = 100
            for k in range(d2.shape[1]):
                hd = utils.hdist(d2[:,j], W2_evects[:,-1-k])
                if hd < min_hdist:
                    min_hdist = hd
            hdists2.append(min_hdist)
        for j in range(d3.shape[1]):
            min_hdist = 100
            for k in range(d3.shape[1]):
                hd = utils.hdist(d3[:,j], W3_evects[:,-1-k])
                if hd < min_hdist:
                    min_hdist = hd
            hdists3.append(min_hdist)
        evect_hdists.append([np.mean(hdists1), np.mean(hdists2), np.mean(hdists3)])

        _,resids1,_,_ = np.linalg.lstsq(d1, W1_evects[:,(-1*d1.shape[1]):])
        _,resids2,_,_ = np.linalg.lstsq(d2, W2_evects[:,(-1*d2.shape[1]):])
        _,resids3,_,_ = np.linalg.lstsq(d3, W3_evects[:,(-1*d3.shape[1]):])
        evect_resids.append([np.mean(resids1), np.mean(resids2), np.mean(resids3)])

        if i < 10:
            print('{}: Solving for fixed points...'.format(i))

            gain = min_pos_e/np.min(W1_evals[W1_evals>1e-7])
            try:
                fxpts1, _ = rfl.run_solver(W1*gain)
            except Exception:
                num_fps.append(None)
                fp_hdists.append(None)
                # fp_lstsq.append(None)
                fp_resids.append(None)
            else:
                num_fps.append(fxpts1.shape[1])

                fp_hd1 = []
                for j in range(d1.shape[1]):
                    min_hdist = 100
                    for k in range(fxpts1.shape[1]):
                        hd = utils.hdist(d1[:,j], fxpts1[:,k])
                        if hd < min_hdist:
                            min_hdist = hd
                    fp_hd1.append(min_hdist)
                fp_hdists.append(fp_hd1)

                lstsq,fp_r1,_,_ = np.linalg.lstsq(d1, fxpts1)
                # fp_lstsq.append(lstsq)
                fp_resids.append(fp_r1)


    return (min_evals_W, min_evals_W_I, num_pos_evals, evect_hdists, evect_resids, num_fps, fp_hdists, fp_resids)

def test_is_data_fp(n, max_evals, orth, zero, scaled):
    if orth and scaled:
        raise ValueError('Cannot use both orthonormal and scaled data')

    dn = open(os.devnull, 'w')

    hdists = []
    resids = []

    hn1 = sch.Hopnet(20)
    hn2 = sch.Hopnet(50)
    hn3 = sch.Hopnet(100)

    for i in range(n):
        if scaled:
            d1 = (np.random.random((20,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(20,3))
            d2 = (np.random.random((50,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(50,3))
            d3 = (np.random.random((100,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(100,3))
        else:
            d1 = np.random.random((20,3))*2-1
            d2 = np.random.random((50,3))*2-1
            d3 = np.random.random((100,3))*2-1
        
            if orth:
                d1 = gd.gs(d1)
                d2 = gd.gs(d2)
                d3 = gd.gs(d3)

        W1 = np.matmul(d1,d1.T)
        W2 = np.matmul(d2,d2.T)
        W3 = np.matmul(d3,d3.T)


        if zero:
            for j in range(W1.shape[0]):
                W1[j,j] = 0
            for j in range(W2.shape[0]):
                W2[j,j] = 0
            for j in range(W3.shape[0]):
                W3[j,j] = 0

        hn1.W = W1
        hn2.W = W2
        hn3.W = W3

        temp_hdists = []
        temp_resids = []
        for v in max_evals:
            gain1 = v/np.max(np.linalg.eigvalsh(W1))
            gain2 = v/np.max(np.linalg.eigvalsh(W2))
            gain3 = v/np.max(np.linalg.eigvalsh(W3))

            hn1.gain = gain1
            hn2.gain = gain2
            hn3.gain = gain3

            j=0
            hn1.simhop(d1[:,j], fileid=dn, trace=False)
            hn2.simhop(d2[:,j], fileid=dn, trace=False)
            hn3.simhop(d3[:,j], fileid=dn, trace=False)

            temp_hdists.append([utils.hdist(hn1.a, d1[:,j]), utils.hdist(hn2.a, d2[:,j]), utils.hdist(hn3.a, d3[:,j])])

            _,resid1,_,_ = np.linalg.lstsq(d1, hn1.a)
            _,resid2,_,_ = np.linalg.lstsq(d2, hn2.a)
            _,resid3,_,_ = np.linalg.lstsq(d3, hn3.a)
            temp_resids.append([resid1[0], resid2[0], resid3[0]])
        hdists.append(temp_hdists)
        resids.append(temp_resids)

    return (np.array(hdists), np.array(resids))


def test_gains(n, max_evals, min_pos_e, orth, zero, scaled):
    if orth and scaled:
        raise ValueError('Cannot use both orthonormal and scaled data')

    dn = open(os.devnull, 'w')

    gains = []


    for i in range(n):
        if scaled:
            d1 = (np.random.random((20,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(20,3))
            d2 = (np.random.random((50,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(50,3))
            d3 = (np.random.random((100,3))/2.0 + 0.5)*np.random.choice([-1,1], size=(100,3))
        else:
            d1 = np.random.random((20,3))*2-1
            d2 = np.random.random((50,3))*2-1
            d3 = np.random.random((100,3))*2-1
        
            if orth:
                d1 = gd.gs(d1)
                d2 = gd.gs(d2)
                d3 = gd.gs(d3)

        W1 = np.matmul(d1,d1.T)
        W2 = np.matmul(d2,d2.T)
        W3 = np.matmul(d3,d3.T)


        if zero:
            for j in range(W1.shape[0]):
                W1[j,j] = 0
            for j in range(W2.shape[0]):
                W2[j,j] = 0
            for j in range(W3.shape[0]):
                W3[j,j] = 0

        W1_evals, W1_evects = np.linalg.eigh(W1)
        W2_evals, W2_evects = np.linalg.eigh(W2)
        W3_evals, W3_evects = np.linalg.eigh(W3)

        temp_gains = []
        for v in max_evals:
            gain1 = v/np.max(W1_evals)
            gain2 = v/np.max(W2_evals)
            gain3 = v/np.max(W3_evals)

            temp_gains.append([gain1, gain2, gain3])

        gain1 = min_pos_e/np.min(W1_evals[W1_evals>1e-7])
        gain2 = min_pos_e/np.min(W2_evals[W2_evals>1e-7])
        gain3 = min_pos_e/np.min(W3_evals[W3_evals>1e-7])
        temp_gains.append([gain1, gain2, gain3])

        gains.append(temp_gains)

    return np.array(gains)


def test_energy_at_memories(n, k, shifted, size=(100,3), max_eval=10.0):
    mem_init_e = []
    mem_fin_e = []
    rand_init_e = []
    rand_fin_e = []

    hn = sch.Hopnet(100)
    dn = open(os.devnull, 'w')

    for i in xrange(n):
        if shifted:
            data = (np.random.random(size)/2.0+0.5)*np.random.choice([-1,1], size=size)
        else:
            data = np.random.random(size)*2-1
        hn.learn(data)
        hn.gain = max_eval/np.max(np.linalg.eigvalsh(hn.W))

        cur_mem_init_e = []
        cur_mem_fin_e = []

        for j in range(size[1]):
            cur_mem_init_e.append(hn.energy(data[:,j], None))
            hn.simhop(data[:,j], fileid=dn, trace=False)
            cur_mem_fin_e.append(hn.energy(hn.a, None))

        cur_rand_init_e = []
        cur_rand_fin_e = []

        for j in range(k):
            if shifted:
                d = (np.random.random((size[0],))/2.0+0.5)*np.random.choice([-1,1], size=(size[0],))
            else:
                d = np.random.random((size[0],))
            cur_rand_init_e.append(hn.energy(d, None))
            hn.simhop(d, fileid=dn, trace=False)
            cur_rand_fin_e.append(hn.energy(hn.a, None))

        mem_init_e.append(cur_mem_init_e)
        mem_fin_e.append(cur_mem_fin_e)
        rand_init_e.append(cur_rand_init_e)
        rand_fin_e.append(cur_rand_fin_e)

    return np.array(mem_init_e), np.array(mem_fin_e), np.array(rand_init_e), np.array(rand_fin_e)


def test_convergence_at_gain(n=1000, size=(100,3), dynamic=False, vals=[1,2,5,10], data_fun=gd.get_random_uniform):
    dists = []

    hn = sch.Hopnet(size[0])
    dn = open(os.devnull, 'w')

    for i in xrange(n):
        data = data_fun(*size)

        hn.learn(data)
        if dynamic:
            max_eval = np.max(np.linalg.eigvalsh(hn.W))
            gains = [v/max_eval for v in vals]
        else:
            gains = vals

        temp_dists = []
        for g in gains:
            hn.gain = g

            sum_dists = 0.0

            for j in xrange(size[1]):
                hn.simhop(data[:,j], fileid=dn, trace=False)
                sum_dists += utils.hdist(data[:,j], hn.a)

            temp_dists.append(sum_dists/size[1])

        dists.append(temp_dists)

    return np.array(dists)


def test_convergence_with_bounds(n=500, size=(100,3), dynamic=True, vals=[1,2,5,10], bounds=[0,0.5,0.99]):
    dists = []

    hn = sch.Hopnet(size[0])
    dn = open(os.devnull, 'w')

    for i in xrange(n):
        dists_bound = []
        for b in bounds:
            utils.wait_until_cool(80)

            data = gd.get_random_gap(*size, bound=b)

            hn.learn(data)
            if dynamic:
                max_eval = np.max(np.linalg.eigvalsh(hn.W))
                gains = [v/max_eval for v in vals]
            else:
                gains = vals

            dists_gain = []
            for g in gains:
                hn.gain = g

                sum_dists = 0.0
                
                for j in xrange(size[1]):
                    hn.simhop(data[:,j], fileid=dn, trace=False)
                    sum_dists += utils.hdist(data[:,j], hn.a)

                dists_gain.append(sum_dists/size[1])

            dists_bound.append(dists_gain)
        dists.append(dists_bound)

    return np.array(dists)

def test_capacity(n=100, size=100, dynamic=True, gain_vals=[2**x for x in range(0,11)], bounds=[0]+[2**x for x in range(-5,0)]+[0.99], err_tol=0.01):
    capacities = []

    hn = sch.Hopnet(size)
    dn = open(os.devnull, 'w')

    for b in bounds:
        

        cap_gain = []
        for v in gain_vals:
            utils.wait_until_cool(80)

            cap = 0
            err = 0

            while err < err_tol:
                utils.wait_until_cool(80)

                cap += 1
                sum_dists = 0
                for i in xrange(n):
                    data = gd.get_random_gap(size,cap, bound=b)

                    hn.learn(data)

                    if dynamic:
                        max_eval = np.max(np.linalg.eigvalsh(hn.W))
                        hn.gain = v/max_eval
                    else:
                        hn.gain = v
                    
                    for j in xrange(cap):
                        hn.simhop(data[:,j], fileid=dn, trace=False)
                        sum_dists += utils.hdist(data[:,j], hn.a)
                err = sum_dists/float(n*size)

            cap -= 1
            cap_gain.append(cap)

        capacities.append(cap_gain)

    return np.array(capacities)


def test_energy_at_fps(n=1, size=(20,3), data_lst=None, gain=5, dynamic=False, traversal=True, dist=utils.hdist):
    dists = []
    energies = []
    fiber_dists = []

    hn = sch.Hopnet(size[0])
    if not dynamic:
        hn.gain = gain

    if data_lst is not None:
        n = len(data_lst)

    for i in xrange(n):
        if data_lst is not None:
            data = data_lst[i]
            size = data.shape
            hn.N = size[0]
        else:
            data = gd.get_random_approx_discrete(*size)

        hn.learn(data)
        if dynamic:
            max_eval = np.max(np.linalg.eigvalsh(hn.W))
            hn.gain = gain/max_eval

        if traversal:
            fps,fiber = rfl.run_solver(hn.W*hn.gain)

            cur_fiber_dists = []
            for j in xrange(fiber.shape[1]):
                # -1 removes last point, a where f(v)=ac on the fiber
                cur_fiber_dists.append([dist(data[:,k], fiber[:-1,j]) for k in xrange(size[1])])
            fiber_dists.append(cur_fiber_dists)

        else:
            fxV, _ = rfl.baseline_solver(hn.W*hn.gain)
            fps, _ = rfl.post_process_fxpts(hn.W*hn.gain, fxV)


        cur_dists = []
        for j in xrange(fps.shape[1]):
            cur_dists.append([dist(data[:,k], fps[:,j]) for k in xrange(size[1])])
        dists.append(cur_dists)
        
        energies.append([hn.energy(fps[:,j], None) for j in xrange(fps.shape[1])])            


    if traversal:
        return np.array(dists), np.array(energies), np.array(fiber_dists)
    else:
        return np.array(dists), np.array(energies)

def test_stability_at_fp(hn, fp, probes=100):
    percent_bits_flipped = []
    avg_dists = []

    dn = open(os.devnull, 'w')

    for i in xrange(hn.N/2+1):
        cur_bf = 0
        cur_sum_dists = 0

        ctr = 0
        for j in xrange(probes):
            probe = np.copy(fp)
            flipped_bits = np.random.choice(xrange(hn.N), size=i, replace=False)
            for bit in flipped_bits:
                probe[bit] *= -1
            hn.simhop(probe, fileid=dn, trace=False)
            cur_bf += utils.hdist(fp, hn.a) # maybe hdist_z?
            cur_sum_dists += np.linalg.norm(fp-hn.a)
            ctr += 1

        percent_bits_flipped.append(cur_bf/float(ctr*hn.N))
        avg_dists.append(cur_sum_dists/float(ctr))

    dn.close()
    return percent_bits_flipped, avg_dists


def test_stability(size=(20,3), data=None, gain=10, traversal=True, probes=100):
    pbfs = []
    dists = []


    if data is not None:
        size = data.shape
    else:
        data = gd.get_random_approx_discrete(size)

    hn = sch.Hopnet(size[0], gain=gain)
    hn.learn(data)

    if traversal:
        fps,_ = rf.run_solver(hn.W*hn.gain)
    else:
        fxV,_ = rf.baseline_solver(hn.W*hn.gain)
        fps,_ = rf.post_process_fxpts(hn.W*hn.gain, fxV)

    for i in xrange(fps.shape[1]):
        pbf,d = test_stability_at_fp(hn, fps[:,i], probes=probes)
        pbfs.append(pbf)
        dists.append(d)

    return fps, pbfs, dists


def test_stability_at_fp_2(hn, fp, probes=100):
    percent_bits_flipped = []
    dists = []

    dn = open(os.devnull, 'w')

    for i in xrange(hn.N/2):
        cur_bf = []
        cur_dists = []

        for j in xrange(probes):
            probe = np.copy(fp)
            flipped_bits = np.random.choice(xrange(hn.N), size=i, replace=False)
            for bit in flipped_bits:
                probe[bit] *= -1
            hn.simhop(probe, fileid=dn, trace=False)
            cur_bf.append(utils.hdist(fp, hn.a)/float(hn.N)) # maybe hdist_z?
            cur_dists.append(np.linalg.norm(fp-hn.a))

        percent_bits_flipped.append(cur_bf)
        dists.append(cur_dists)

    dn.close()
    return np.array(percent_bits_flipped), np.array(dists)


def test_stability_2(size=(20,3), data=None, gain=10, traversal=True, probes=100):
    pbfs = []
    dists = []


    if data is not None:
        size = data.shape
    else:
        data = gd.get_random_approx_discrete(size)

    hn = sch.Hopnet(size[0], gain=gain)
    hn.learn(data)

    if traversal:
        fps,_ = rf.run_solver(hn.W*hn.gain)
    else:
        fxV,_ = rf.baseline_solver(hn.W*hn.gain)
        fps,_ = rf.post_process_fxpts(hn.W*hn.gain, fxV)

    for i in xrange(fps.shape[1]):
        pbf,d = test_stability_at_fp_2(hn, fps[:,i], probes=probes)
        pbfs.append(pbf)
        dists.append(d)

    return fps, pbfs, dists

def test_stability_at_mems(n=10, size=100, gain=10, num_mems=[1,2,4,8,16,32], num_bits_to_flip=[0,1,2,4,8,16,32,50], probes=50):
    pbfs = []
    dists = []

    dn = open(os.devnull, 'w')
    hn = sch.Hopnet(size, gain=gain)

    for i in xrange(n):
        cur_pbfs = np.zeros((len(num_mems),len(num_bits_to_flip)))
        cur_dists = np.zeros((len(num_mems),len(num_bits_to_flip)))
        for j,nm in enumerate(num_mems):
            for k,nbtf in enumerate(num_bits_to_flip):
                data = gd.get_random_approx_discrete(size,nm)
                hn.learn(data)

                for mem in xrange(nm):
                    for l in xrange(probes):
                        probe = np.copy(data[:,mem])
                        flipped_bits = np.random.choice(xrange(size), size=nbtf, replace=False)
                        for bit in flipped_bits:
                            probe[bit] *= -1
                        hn.simhop(probe, fileid=dn, trace=False)
                        cur_pbfs[j,k] += utils.hdist(data[:,mem], hn.a) # maybe hdist_z?
                        cur_dists[j,k] += np.linalg.norm(data[:,mem]-hn.a)
                cur_pbfs[j,k] /= float(size*nm*probes)
                cur_dists[j,k] /= float(nm*probes)
        pbfs.append(cur_pbfs)
        dists.append(cur_dists)

    dn.close()

    return np.array(pbfs), np.array(dists)


def test_async_fps(n=10, size=(20,3), gain=5, traversal=True, tol=1e-8):
    # dists = []
    # hdists = []
    bad_data = []

    dn = open(os.devnull, 'w')
    hn = ach.Hopnet(size[0], gain=gain)

    for i in xrange(n):
        # cur_dists = []
        # cur_hdists = []

        data = gd.get_random_approx_discrete(*size)

        hn.learn(data)

        if traversal:
            fps,_ = rf.run_solver(hn.W*hn.gain)
        else:
            fxV,_ = rf.baseline_solver(hn.W*hn.gain)
            fps,_ = rf.post_process_fxpts(hn.W*hn.gain, fxV)

        for j in xrange(fps.shape[1]):
            hn.simhop(fps[:,j], fileid=dn, trace=False, tolerance=tol)
            if not np.allclose(hn.a, fps[:,j], rtol=0, atol=tol):
                bad_data.append(data)
                break
        #     cur_dists.append(np.linalg.norm(hn.a-fps[:,j]))
        #     cur_hdists.append(utils.hdist(hn.a,fps[:,j])) # Maybe hdist_z?

        # dists.append(cur_dists)
        # hdists.append(cur_hdists)

    # return np.array(hdists),np.array(dists)
    return bad_data


def test_jacobian(n=10, sync=True, size=(20,3), gain=5, traversal=True, perturbation=0.01, probes=10, tol=1e-6):
    bad_res = [] # [(data, fps, J_eigs, probes, deviations, index)]

    dn = open(os.devnull, 'w')
    if sync:
        hn = sch.Hopnet(size[0], gain=gain)
    else:
        hn = ach.Hopnet(size[0], gain=gain)

    for i in xrange(n):

        data = gd.get_random_approx_discrete(*size)

        hn.learn(data)

        if traversal:
            fps,_ = rf.run_solver(hn.W*hn.gain)
        else:
            fxV,_ = rf.baseline_solver(hn.W*hn.gain)
            fps,_ = rf.post_process_fxpts(hn.W*hn.gain, fxV)

        for j in xrange(fps.shape[1]):
            # J_eigs = np.real(np.linalg.eigvals(hn.jacobian(fps[:,j])))
            # J_eig_neg =  np.all(1>np.abs(J_eigs)) #np.all(0>J_eigs)
            J_eigs = np.linalg.eigvals(hn.jacobian(fps[:,j]))
            J_eig_stable = np.all(1>np.absolute(J_eigs))
            deviations = 0
            bad_probes = []
            for p in xrange(probes):
                probe = fps[:,j] + perturbation*np.random.choice([-1,1], size[0])
                hn.simhop(probe, fileid=dn, trace=False, tolerance=tol/10)
                if J_eig_stable:
                    if not np.allclose(hn.a, fps[:,j], rtol=0, atol=tol):
                        deviations += 1
                        bad_probes.append(probe)
                else:
                    if np.allclose(hn.a, fps[:,j], rtol=0, atol=tol):
                        deviations -= 1
                        bad_probes.append(probe)
            if deviations != 0:
                bad_res.append((data,fps,J_eigs,bad_probes,deviations,j))

    return bad_res

def test_stability_at_fp_3(hn, fp, probes=100, perturbation=0.01, tol=1e-6):
    deviations = 0
    conv_p = []
    div_p = []

    dn = open(os.devnull, 'w')

    J_eigs = np.real(np.linalg.eigvals(hn.jacobian(fp)))
    J_eig_neg = np.all(0>J_eigs)

    for j in xrange(probes):
        probe = fp + perturbation*np.random.choice([-1,1], fp.shape[0])
        hn.simhop(probe, fileid=dn, trace=False, tolerance=tol/10)
        if J_eig_neg:
            if not np.allclose(hn.a, fp, rtol=0, atol=tol):
                deviations += 1
                div_p.append(probe)
            else:
                conv_p.append(probe)
        else:
            if np.allclose(hn.a, fp, rtol=0, atol=tol):
                deviations -= 1
                conv_p.append(probe)
            else:
                div_p.append(probe)


    dn.close()
    return deviations,np.array(conv_p).T,np.array(div_p).T


def test_stability_3(size=(20,3), data=None, gain=5, traversal=True, probes=100, perturbation=0.01, tol=1e-6):
    dev_lst = []


    if data is not None:
        size = data.shape
    else:
        data = gd.get_random_approx_discrete(*size)

    hn = sch.Hopnet(size[0], gain=gain)
    hn.learn(data)

    if traversal:
        fps,_ = rf.run_solver(hn.W*hn.gain)
    else:
        fxV,_ = rf.baseline_solver(hn.W*hn.gain)
        fps,_ = rf.post_process_fxpts(hn.W*hn.gain, fxV)

    for i in xrange(fps.shape[1]):
        d,_,_ = test_stability_at_fp_3(hn, fps[:,i], probes=probes, perturbation=perturbation, tol=tol)
        dev_lst.append(d)
        
    return fps,dev_lst

def test_compare_solvers(n=10, size=100, gain=10, mems=[2*i+1 for i in xrange(9)], sync=True):
    baseline_stable = []
    baseline_unstable = []
    traversal_stable = []
    traversal_unstable = []

    if sync:
        hn = sch.Hopnet(size, gain=gain)
    else:
        hn = ach.Hopnet(size, gain=gain)

    for m in mems:
        cur_bs = 0
        cur_bu = 0
        cur_ts = 0
        cur_tu = 0

        for i in xrange(n):
            data = gd.get_random_approx_discrete(size,m)
            hn.learn(data)

            # Baseline:
            fxV,_ = rf.baseline_solver(hn.W*hn.gain)
            fps_b,_ = rf.post_process_fxpts(hn.W*hn.gain, fxV)

            for j in xrange(fps_b.shape[1]):
                J_eigs = np.real(np.linalg.eigvals(hn.jacobian(fps_b[:,j])))
                if np.all(1>np.abs(J_eigs)): #np.all(0>J_eigs)
                    cur_bs += 1
                else:
                    cur_bu += 1

            # Traversal:
            fps_t,_ = rf.run_solver(hn.W*hn.gain)

            for j in xrange(fps_t.shape[1]):
                J_eigs = np.real(np.linalg.eigvals(hn.jacobian(fps_t[:,j])))
                if np.all(1>np.abs(J_eigs)): #np.all(0>J_eigs)
                    cur_ts += 1
                else:
                    cur_tu += 1

        baseline_stable.append(cur_bs/float(n))
        baseline_unstable.append(cur_bu/float(n))
        traversal_stable.append(cur_ts/float(n))
        traversal_unstable.append(cur_tu/float(n))

    return baseline_stable,baseline_unstable,traversal_stable,traversal_unstable

def test_invertibility(n=1000, size=20, mems=[1], tol=1e-8):
    bad_data = []

    for m in mems:
        cur_bad_data = []

        for i in xrange(n):

            data = gd.get_random_approx_discrete(size,m)
            W = np.matmul(data,data.T)
            for i in range(size):
                W[i,i] = 0
            W *= (1.0/size)

            if np.min(np.abs(np.linalg.eigvalsh(W)))<tol:
                cur_bad_data.append(data)

        bad_data.append(cur_bad_data)

    return bad_data

def test_traversal_along_evects(n=10, size=(20,3), gain=10, sync=True, probes=5):
    def get_c_from_evect(hn, evect, delta=0.01):
        a_init = evect*delta
        hn.a = a_init
        hn.update()
        return (hn.a-a_init)/np.linalg.norm(hn.a-a_init)


    random_stable = []
    random_unstable = []
    evects_stable = []
    evects_unstable = []

    if sync:
        hn = sch.Hopnet(size[0], gain=gain)
    else:
        hn = ach.Hopnet(size[0], gain=gain)

    for i in xrange(n):
        data = gd.get_random_approx_discrete(*size)
        hn.learn(data)

        # Random
        cur_rs = [0]*probes
        cur_ru = [0]*probes
        for p in xrange(probes):
            fps,_ = rf.run_solver(hn.W*hn.gain)

            for j in xrange(fps.shape[1]):
                # J_eigs = np.real(np.linalg.eigvals(hn.jacobian(fps[:,j])))
                # if np.all(1>np.abs(J_eigs)): #np.all(0>J_eigs)
                if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                    cur_rs[p] += 1
                else:
                    cur_ru[p] += 1
        random_stable.append(cur_rs)
        random_unstable.append(cur_ru)

        # Eigenvectors
        _,evects = np.linalg.eigh(hn.W*hn.gain)

        cur_es = [0]*size[1]
        cur_eu = [0]*size[1]
        for e in xrange(size[1]):
            fps,_ = rf.run_solver(hn.W*hn.gain, c=get_c_from_evect(evects[:,-1-e]).reshape(-1,1))

            for j in xrange(fps.shape[1]):
                # J_eigs = np.real(np.linalg.eigvals(hn.jacobian(fps[:,j])))
                # if np.all(1>np.abs(J_eigs)): #np.all(0>J_eigs)
                if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                    cur_es[e] += 1
                else:
                    cur_eu[e] += 1
        evects_stable.append(cur_es)
        evects_unstable.append(cur_eu)

    return np.array(random_stable), np.array(random_unstable), np.array(evects_stable), np.array(evects_unstable)

def solver_wrapper(args):
    n, traversal, W = args
    if traversal:
        # Runs solver n times with different c
        fxpts = []
        for i in xrange(n):
            for iterate in rf.directional_fiber(W, c=None, max_traverse_steps = 2**20):
                fxpts.append(iterate[1])
        fxpts = np.concatenate(fxpts, axis=1)
        fxpts, _ = rf.post_process_fxpts(W, fxpts)
    else:
        fxV, _ = rf.baseline_solver(W)
        for i in xrange(n-1):
            cur_fxV, _ = rf.baseline_solver(W)
            np.concatenate([fxV,cur_fxV], axis=1)
        fxpts, _ = rf.post_process_fxpts(W, fxV)
    return fxpts

def test_fp_varying_gain(data, dynamic=True, gains=[0.1,0.9,1.1,10,100,1000,10000], traversal=True, probes=5):
    hn = sch.Hopnet(data.shape[0])
    hn.learn(data)

    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        gains = [g/max_eval for g in gains]

    args = [(probes, traversal, hn.W*g) for g in gains]

    pool = mp.Pool(min(len(gains), 16))
    fps = pool.map(solver_wrapper, args)

    fps_count = []

    for fps_lst in fps:
        fps_count.append(fps_lst.shape[1])

    # for g in gains:
    #     cur_fps = solver_wrapper(probes, hn.W*g)
    #     fps.append(cur_fps)
    #     fps_count.append(cur_fps.shape[1])

    return fps, fps_count, gains


def jacobian_wrapper(args):
    probe,hn,tol = args

    J = hn.jacobian(np.array(probe))
    e_min = spl.eigh(J.T.dot(J), eigvals_only=True, eigvals=(0,1))[0]
    if e_min < tol:
        # Find associated c,a
        # f(probe) = a*c for some a, unit c
        v = np.arctanh(probe)
        c = v/np.linalg.norm(v)
        a = np.linalg.norm(v)
        return(c,a)
    else:
        return None

def jacobian_wrapper2(args):
    probe,hn = args

    J = hn.jacobian(np.array(probe))
    e_min = spl.eigh(J.T.dot(J), eigvals_only=True, eigvals=(0,1))[0]
    
    return probe,e_min

def test_find_critical_pts(data, gain, grid_1d, sync=True, tol=1e-3, compute_c=True):
    if sync:
        hn = sch.Hopnet(data.shape[0], gain=gain)
    else:
        hn = ach.Hopnet(data.shape[0], gain=gain)
    hn.learn(data)

    prods = it.product(grid_1d, repeat=data.shape[0])

    pool = mp.Pool(16)
    if compute_c:
        res_lst = pool.map(jacobian_wrapper, it.product(prods, [hn], [tol]), chunksize=64)
        return list(it.ifilter(lambda x: x is not None, res_lst))
    else:
        res_lst = pool.map(jacobian_wrapper2, it.product(prods, [hn]), chunksize=64)
        return res_lst


def test_solver_wrapper(args):
    c,W = args
    try:
        _,_,e_mins = rf.run_solver(W, c=np.array(c).reshape(-1,1))
        return np.min(e_mins)
    except Exception, arg:
        print(arg)
        return 0

def test_check_scpf_results(data, gain=1.0, dynamic=False, all_orthants=False):
    # Note: assumes rnn_fxpts has been modified to return list of e_mins
    assert data.shape[0] <= 26

    min_e_mins = []

    hn = sch.Hopnet(data.shape[0])
    hn.learn(data)

    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        hn.gain = gain/max_eval
    else:
        hn.gain = gain

    variables = sp.symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z', real=True)[:data.shape[0]]
    
    soln = scpf.get_soln_for_var(hn.W, variables, variables[0], hn.gain)
    soln_lambda = scpf.lambdify_soln(soln, variables, variables[0])
    solns = scpf.get_solns_in_grid(soln_lambda, variables, variables[0])
    v_lst,c_lst,a_lst = scpf.solns_to_vca(solns, hn.W, hn.gain, all_orthants=all_orthants)

    pool = mp.Pool(8)
    min_e_mins = pool.map(test_solver_wrapper, it.product(c_lst, [hn.W*hn.gain]), chunksize=64)

    return v_lst,c_lst,a_lst,min_e_mins

def test_scpf_full(n, shape, gain=1.0, dynamic=False):
    incorrect_v = []
    incorrect_c = []
    incorrect_a = []
    incorrect_min_e_mins = []

    for i in xrange(n):
        data = gd.get_random_uniform(*shape)

        v_lst,c_lst,a_lst,min_e_mins = test_check_scpf_results(data, gain=gain, dynamic=dynamic)

        if np.max(min_e_mins) > 0:
            incorrect_v.append(v_lst)
            incorrect_c.append(c_lst)
            incorrect_a.append(a_lst)
            incorrect_min_e_mins.append(min_e_mins)

    return incorrect_v,incorrect_c,incorrect_a,incorrect_min_e_mins


def test_standard_vs_cont_hopnet(data, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-16):
    standard_fps = []
    cont_fps = []

    hn = chc.Hopnet(data.shape[0])
    hn.learn(data)

    # Find fixed points of standard Hopnet
    for p in it.product([-1.0,1.0], repeat=data.shape[0]):
        probe = np.array(p)
        if np.allclose(np.sign(np.dot(hn.W,probe)), probe, rtol=0, atol=tol): # What about 0?
            standard_fps.append(probe)

    matched_fps = np.zeros((len(standard_fps),len(gains)), dtype=np.int_) # [[0]*len(gains)]*len(standard_fps)
    unmatched_fps = np.zeros(len(gains), dtype=np.int_) # [0]*len(gains)

    # Find fixed points of cont Hopnet at various gain values
    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        gains_ = [g/max_eval for g in gains]
    else:
        gains_ = gains


    for i,g in enumerate(gains_):
        if traversal:
            fps,_ = rf.run_solver(hn.W*g)

        else:
            fxV, _ = rfl.baseline_solver(hn.W*g)
            fps, _ = rfl.post_process_fxpts(hn.W*g, fxV)

        cont_fps.append(fps)

        sgn_fps = np.sign(fps)

        # TODO: check stability?
        for j in xrange(sgn_fps.shape[1]):
            for k,fp in enumerate(standard_fps):
                if np.allclose(sgn_fps[:,j], fp, rtol=0, atol=tol):
                    matched_fps[k][i] += 1
                    break
            else:
                unmatched_fps[i] += 1

    return standard_fps,cont_fps,matched_fps,unmatched_fps

def test_s_vs_c_hn_full(n, shape, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-6):
    matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)

    for i in xrange(n):
        data = gd.get_random_discrete(*shape)
        _,_,m_fps,um_fps = test_standard_vs_cont_hopnet(data, gains, dynamic, traversal, tol)
        matched_fps[i] += np.sum(m_fps,axis=0)
        unmatched_fps[i] += um_fps

    return matched_fps,unmatched_fps

def test_standard_vs_jr_euclid(data, stepsize=0.1, ini_mult=0.5, tol=1e-8, max_steps=500):
    problem_probes = []
    jr_prob_res = []
    st_prob_res = []

    def update_jr_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            res[i] += gain*np.dot(W[i,:],res)*(1-res[i]**2)
        return res
    def update_standard_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], res)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanged
        return res
    def update_jr_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            res[i] += gain*np.dot(W[i,:],v)*(1-v[i]**2)
        return res
    def update_standard_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], v)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanges
        return res

    hn_jr = chc.Hopnet(data.shape[0], mode=(update_jr_async,None,None), gain=stepsize)
    hn_jr.learn(data)

    hn_st = chc.Hopnet(data.shape[0], mode=(update_standard_async,None,None))
    hn_st.learn(data)

    for p in it.product([-1.0,1.0], repeat=data.shape[0]):
        probe = np.array(p, dtype=np.float32)
        
        _,res_jr,_ = hn_jr.simhop(ini_mult*probe, silent=True, tolerance=tol, max_steps=max_steps)
        _,res_st,_ = hn_st.simhop(probe, silent=True, tolerance=tol, max_steps=max_steps)

        if not np.allclose(np.sign(res_st), np.sign(res_jr), rtol=0, atol=1e-16):
            problem_probes.append(probe)
            jr_prob_res.append(res_jr)
            st_prob_res.append(res_st)

    return problem_probes,jr_prob_res,st_prob_res

def test_standard_vs_jr_spi_ode(data, ini_mult=0.5, t=50):
    problem_probes = []
    jr_prob_res = []
    st_prob_res = []

    def update_standard_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], res)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanged
        return res
    def update_standard_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], v)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanges
        return res
    def dvdt(v, t, W):
        return np.dot(W,v)*(1-v**2)


    hn_st = chc.Hopnet(data.shape[0], mode=(update_standard_sync,None,None))
    hn_st.learn(data)

    time = np.linspace(0, t, t*10)

    for p in it.product([-1.0,1.0], repeat=data.shape[0]):
        probe = np.array(p, dtype=np.float32)
        
        ode_res = spi.odeint(dvdt, probe*ini_mult, time, args=(hn_st.W,))
        res_jr = ode_res[-1,:]
        _,res_st,_ = hn_st.simhop(probe, silent=True)

        if not np.allclose(np.sign(res_st), np.sign(res_jr), rtol=0, atol=1e-16):
            problem_probes.append(probe)
            jr_prob_res.append(res_jr)
            st_prob_res.append(res_st)

    return problem_probes,jr_prob_res,st_prob_res

def test_standard_vs_cont_stability(data, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-16):
    standard_fps = []
    cont_fps = []

    hn = chc.Hopnet(data.shape[0])
    hn.learn(data)

    # Find fixed points of standard Hopnet
    for p in it.product([-1.0,1.0], repeat=data.shape[0]):
        probe = np.array(p)
        if np.allclose(np.sign(np.dot(hn.W,probe)), probe, rtol=0, atol=tol): # What about 0?
            standard_fps.append(probe)

    stable_matched_fps = np.zeros((len(standard_fps),len(gains)), dtype=np.int_) # [[0]*len(gains)]*len(standard_fps)
    unstable_matched_fps = np.zeros((len(standard_fps),len(gains)), dtype=np.int_) # [[0]*len(gains)]*len(standard_fps)
    stable_unmatched_fps = np.zeros(len(gains), dtype=np.int_) # [0]*len(gains)
    unstable_unmatched_fps = np.zeros(len(gains), dtype=np.int_) # [0]*len(gains)

    # Find fixed points of cont Hopnet at various gain values
    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        gains_ = [g/max_eval for g in gains]
    else:
        gains_ = gains


    for i,g in enumerate(gains_):
        if traversal:
            fps,_ = rf.run_solver(hn.W*g)

        else:
            fxV, _ = rfl.baseline_solver(hn.W*g)
            fps, _ = rfl.post_process_fxpts(hn.W*g, fxV)

        cont_fps.append(fps)

        sgn_fps = np.sign(fps)

        
        for j in xrange(sgn_fps.shape[1]):
            if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                for k,fp in enumerate(standard_fps):
                    if np.allclose(sgn_fps[:,j], fp, rtol=0, atol=tol):
                        stable_matched_fps[k][i] += 1
                        break
                else:
                    stable_unmatched_fps[i] += 1
            else:
                for k,fp in enumerate(standard_fps):
                    if np.allclose(sgn_fps[:,j], fp, rtol=0, atol=tol):
                        unstable_matched_fps[k][i] += 1
                        break
                else:
                    unstable_unmatched_fps[i] += 1

    return standard_fps,cont_fps,stable_matched_fps,unstable_matched_fps,stable_unmatched_fps,unstable_unmatched_fps

def test_s_vs_c_stability_wrapper(args):
    shape, gains, dynamic, traversal, tol = args
    np.random.seed()
    data = gd.get_random_discrete(*shape)
    _,_,sm,um,su,uu = test_standard_vs_cont_stability(data, gains, dynamic, traversal, tol)
    return np.sum(sm,axis=0),np.sum(um,axis=0),su,uu

def test_s_vs_c_stability_full(n, shape, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-6):
    stable_matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    stable_unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)

    # for i in xrange(n):
    #     data = gd.get_random_discrete(*shape)
    #     _,_,sm,um,su,uu = test_standard_vs_cont_stability(data, gains, dynamic, traversal, tol)
    #     stable_matched_fps[i] += np.sum(sm,axis=0)
    #     unstable_matched_fps[i] += np.sum(um,axis=0)
    #     stable_unmatched_fps[i] += su
    #     unstable_unmatched_fps[i] += uu
    pool = mp.Pool(16)
    res = pool.map(test_s_vs_c_stability_wrapper, [(shape,gains,dynamic,traversal,tol)]*n, chunksize=2)

    for i in xrange(n):
        stable_matched_fps[i] += res[i][0]
        unstable_matched_fps[i] += res[i][1]
        stable_unmatched_fps[i] += res[i][2]
        unstable_unmatched_fps[i] += res[i][3]


    return stable_matched_fps,unstable_matched_fps,stable_unmatched_fps,unstable_unmatched_fps

def test_standard_vs_jr_euclid2(data, stepsize=0.1, ini_mult=0.5, tol=1e-8, max_steps=500, jr_sync=False, st_sync=False):
    problem_probes = []
    jr_prob_res = []
    st_prob_res = []

    def update_jr_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            res[i] += gain*np.dot(W[i,:],res)*(1-res[i]**2)
        return res
    def update_standard_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], res)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanged
        return res
    def update_jr_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            res[i] += gain*np.dot(W[i,:],v)*(1-v[i]**2)
        return res
    def update_standard_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], v)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanged
        return res

    if jr_sync:
        hn_jr = chc.Hopnet(data.shape[0], mode=(update_jr_sync,None,None), gain=stepsize)
    else:
        hn_jr = chc.Hopnet(data.shape[0], mode=(update_jr_async,None,None), gain=stepsize)
    hn_jr.learn(data)

    if st_sync:
        hn_st = chc.Hopnet(data.shape[0], mode=(update_standard_sync,None,None))
    else:
        hn_st = chc.Hopnet(data.shape[0], mode=(update_standard_async,None,None))
    hn_st.learn(data)

    for i in xrange(data.shape[1]):
        for j in xrange(data.shape[0]):
            probe = np.copy(data[:,i])
            probe[j] *= -1.0

            _,res_jr,_ = hn_jr.simhop(ini_mult*probe, silent=True, tolerance=tol, max_steps=max_steps)
            _,res_st,_ = hn_st.simhop(probe, silent=True, tolerance=tol, max_steps=max_steps)

            if not np.allclose(np.sign(res_st), np.sign(res_jr), rtol=0, atol=1e-16):
                problem_probes.append(probe)
                jr_prob_res.append(res_jr)
                st_prob_res.append(res_st)

    return problem_probes,jr_prob_res,st_prob_res

def test_standard_vs_jr_spi_ode2(data, ini_mult=0.5, t=50):
    problem_probes = []
    jr_prob_res = []
    st_prob_res = []

    def update_standard_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], res)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanged
        return res
    def update_standard_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], v)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanges
        return res
    def dvdt(v, t, W):
        return np.dot(W,v)*(1-v**2)


    hn_st = chc.Hopnet(data.shape[0], mode=(update_standard_async,None,None))
    hn_st.learn(data)

    time = np.linspace(0, t, t*10)

    for i in xrange(data.shape[1]):
        for j in xrange(data.shape[0]):
            probe = np.copy(data[:,i])
            probe[j] *= -1.0
        
            ode_res = spi.odeint(dvdt, probe*ini_mult, time, args=(hn_st.W,))
            res_jr = ode_res[-1,:]
            _,res_st,_ = hn_st.simhop(probe, silent=True)

            if not np.allclose(np.sign(res_st), np.sign(res_jr), rtol=0, atol=1e-16):
                problem_probes.append(probe)
                jr_prob_res.append(res_jr)
                st_prob_res.append(res_st)

            for k in xrange(data.shape[0]):
                probe[k] *= -1.0 # Bad - leaves old bits flipped
            
                ode_res = spi.odeint(dvdt, probe*ini_mult, time, args=(hn_st.W,))
                res_jr = ode_res[-1,:]
                _,res_st,_ = hn_st.simhop(probe, silent=True)

                if not np.allclose(np.sign(res_st), np.sign(res_jr), rtol=0, atol=1e-16):
                    problem_probes.append(probe)
                    jr_prob_res.append(res_jr)
                    st_prob_res.append(res_st)

    return problem_probes,jr_prob_res,st_prob_res


def jr_ode_content_addr_wrapper(args):
    def dvdt(v, t, W):
        return np.dot(W,v)*(1-v**2)

    shape,mems,ini_mults,coords_dist,probes,time = args
    hn = chc.Hopnet(shape)

    cur_num_correct = np.zeros((len(coords_dist),len(mems),len(ini_mults)), dtype=np.float32)

    for j,m in enumerate(mems):
        data = gd.get_random_discrete(shape,m)
        hn.learn(data)
        for k in xrange(m):
            for l,cd in enumerate(coords_dist):
                for _ in xrange(probes):
                    probe = np.copy(data[:,k])
                    for rand in np.random.choice(shape, size=cd, replace=False):
                        probe[rand] *= -1.0

                    for q,im in enumerate(ini_mults):
                        ode_res = spi.odeint(dvdt, probe*im, time, args=(hn.W,))
                        res_jr = ode_res[-1,:]

                        if np.all(np.sign(res_jr) - np.sign(data[:,k]) < 1e-16):
                            cur_num_correct[l,j,q] += 1.0/(m*probes)

    return cur_num_correct


def test_jr_ode_content_addr(n, shape, mems=[i+1 for i in xrange(10)], ini_mults=[(i+1)*0.1 for i in xrange(9)], coords_dist=[1,2,3], probes=10, t=50):
    

    # num_correct = np.zeros((len(coords_dist),len(mems),len(ini_mults)), dtype=np.float32)

    time = np.linspace(0, t, t*10)

    # hn = chc.Hopnet(shape)

    # for i in xrange(n):
    #     cur_num_correct = np.zeros(num_correct.shape, dtype=np.float32)

    #     for j,m in enumerate(mems):
    #         data = gd.get_random_discrete(shape,m)
    #         hn.learn(data)
    #         for k in xrange(m):
    #             for l,cd in enumerate(coords_dist):
    #                 for _ in xrange(probes):
    #                     probe = np.copy(data[:,k])
    #                     for _ in xrange(cd):
    #                         probe[np.random.randint(m)] *= -1.0

    #                     for q,im in enumerate(ini_mults):
    #                         ode_res = spi.odeint(dvdt, probe*im, time, args=(hn.W,))
    #                         res_jr = ode_res[-1,:]

    #                         if np.all(np.sign(res_jr) - np.sign(data[:,k]) < 1e-16):
    #                             cur_num_correct[l,j,q] += 1.0/(m*probes)

    #     num_correct += cur_num_correct

    args = [(shape,mems,ini_mults,coords_dist,probes,time)]*n

    pool = mp.Pool(min(n, 16))
    res = pool.map(jr_ode_content_addr_wrapper, args)

    return np.sum(res, axis=0) * 1.0/n


def standard_vs_jr_spi_ode_wrapper(args):
    def update_standard_async(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], res)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanged
        return res
    def update_standard_sync(W,gain,v):
        res = np.copy(v)
        for i in xrange(res.shape[0]):
            Wv_i = np.dot(W[i,:], v)
            if Wv_i > 0:
                res[i] = 1.0
            elif Wv_i < 0:
                res[i] = -1.0
            # else unchanges
        return res
    def dvdt(v, t, W):
        return np.dot(W,v)*(1-v**2)

    shape,mems,ini_mults,probes,time = args
    hn = chc.Hopnet(shape, mode=(update_standard_async,None,None))

    cur_num_match = np.zeros((len(mems),len(ini_mults)), dtype=np.float32)

    for i in xrange(len(mems)):
        for j in xrange(len(ini_mults)):
            data = gd.get_random_discrete(shape,mems[i])
            hn.learn(data)
            for _ in xrange(probes):
                probe = gd.get_random_discrete(shape,1)[:,0]

                ode_res = spi.odeint(dvdt, probe*ini_mults[j], time, args=(hn.W,))
                res_jr = ode_res[-1,:]
                _,res_st,_ = hn.simhop(probe, silent=True)

                if np.allclose(np.sign(res_st), np.sign(res_jr), rtol=0, atol=1e-16):
                    cur_num_match[i,j] += 1.0/probes

    return cur_num_match
                


def test_standard_vs_jr_spi_ode_full(n, shape, mems=[i+1 for i in xrange(15)], ini_mults=[(i+1)*0.1 for i in xrange(9)], probes=100, t=50):
    num_match = np.zeros((len(mems),len(ini_mults)), dtype=np.float32)

    time = np.linspace(0, t, t*10)

    args = [(shape,mems,ini_mults,probes,time)]*n

    pool = mp.Pool(min(n, 16))
    res = pool.map(standard_vs_jr_spi_ode_wrapper, args)

    return np.sum(res, axis=0) * 1.0/n


def test_sample_cont_time_fps(n, data):
    """Test running solver with varying c on cont-time Hopnet"""

    fps = []
    fps_edists = []
    fps_hdists = []
    sanitized_fps = []
    kwargs = cth.get_solver_kwargs(data)
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    for i in xrange(n):
        soln = df_sv.fiber_solver(**kwargs)
        fps.append(soln['Fixed points'])

    try:
        sanitized_fps = df_fx.sanitize_points(np.concatenate(fps,axis=1), kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)
    except Exception, arg:
        print('Could not sanitize fixed points:', arg)
        sanitized_fps = np.concatenate(fps, axis=1)

    # fps_compressed = np.concatenate(sanitized_fps, axis=1)

    z = np.zeros(data.shape[0])

    for d in xrange(data.shape[1]):
        fps_edists.append(np.array([np.linalg.norm(data[:,d]-sanitized_fps[:,i]) for i in xrange(sanitized_fps.shape[1])]))
        fps_hdists.append(np.array([utils.hdist(data[:,d],sanitized_fps[:,i]) for i in xrange(sanitized_fps.shape[1]) 
            if not np.allclose(sanitized_fps[:,i],z)]))

    return sanitized_fps,fps_edists,fps_hdists


def test_cont_time_stability(n, shape, step_size=1e-1):
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    stable = 0
    unstable = 0
    stable_fps = []
    unstable_fps = []
    data_lst = []
    kwargs_lst = []

    for i in xrange(n):
        print(i)
        
        cur_stable = []
        cur_unstable = []

        data = gd.get_random_discrete(*shape)
        kwargs = cth.get_solver_kwargs(data, step_size=step_size)

        soln = df_sv.fiber_solver(**kwargs)
        fps = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)
        
        for j in xrange(fps.shape[1]):
            if np.all(0>np.real(np.linalg.eigvals(kwargs['Df'](fps[:,j][:,np.newaxis])))):
                stable += 1
                cur_stable.append(fps[:,j])
            else:
                unstable += 1
                cur_unstable.append(fps[:,j])

        stable_fps.append(cur_stable)
        unstable_fps.append(cur_unstable)
        data_lst.append(data)
        kwargs_lst.append(kwargs)

    return stable,unstable,stable_fps,unstable_fps,data_lst,kwargs_lst

def test_cont_time_f_at_fps(n, shape):
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)
    max_f = []

    for i in xrange(n):
        data = gd.get_random_discrete(*shape)
        kwargs = cth.get_solver_kwargs(data)

        soln = df_sv.fiber_solver(**kwargs)
        fps = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)

        for j in xrange(fps.shape[1]):
            max_f.append(np.amax(kwargs['f'](fps[:,j])))

    return max_f

def test_cont_time_c_vs_fps(n, data):
    fps = []
    c_lst = []
    data_c_hdists = []
    data_fps_hdists = []


    kwargs = cth.get_solver_kwargs(data)
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    for i in xrange(n):
        c = np.random.randn(data.shape[0],1)
        c = c/np.sqrt((c**2).sum())
        c_lst.append(c)

        kwargs['c'] = c
        soln = df_sv.fiber_solver(**kwargs)
        fps.append(df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates))


    for i in xrange(len(fps)):
        cur_data_c_hdists = []
        cur_data_fps_hdists = []

        for j in xrange(data.shape[1]):
            cur_data_c_hdists.append(utils.hdist(data[:,j], c_lst[i][:,0]))

            cur_data_fps_hdists_2 = []

            for k in xrange(fps[i].shape[1]):
                cur_data_fps_hdists_2.append(utils.hdist(data[:,j], fps[i][:,k]))

            cur_data_fps_hdists.append(cur_data_fps_hdists_2)

        data_c_hdists.append(cur_data_c_hdists)
        data_fps_hdists.append(cur_data_fps_hdists)

    return fps,c_lst,data_c_hdists,data_fps_hdists

def test_step_size_cont_time_fps(data, c=None, steps=[1e-1,1e-3,1e-5,1e-7,1e-9]):
    if c is None:
        c = np.random.randn(data.shape[0],1)
        c = c/np.sqrt((c**2).sum())

    fps = []
    fps_edists = []
    fps_hdists = []
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    z = np.zeros(data.shape[0])

    for step in steps:
        kwargs = cth.get_solver_kwargs(data, step_size=step)
        kwargs['c'] = c
        soln = df_sv.fiber_solver(**kwargs)
        cur_fps = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)
        fps.append(cur_fps)

        fps_edists.append(np.array([[np.linalg.norm(data[:,d]-cur_fps[:,i]) for i in xrange(cur_fps.shape[1])] for d in xrange(data.shape[1])]))
        fps_hdists.append(np.array([[utils.hdist(data[:,d],cur_fps[:,i]) for i in xrange(cur_fps.shape[1]) 
            if not np.allclose(cur_fps[:,i],z)] for d in xrange(data.shape[1])]))


    return fps,fps_edists,fps_hdists


def test_step_size_cont_time_fps_many_c(n, data, steps=[1e-1,1e-3,1e-5,1e-7,1e-9]):

    fps = []
    fps_edists = []
    fps_hdists = []
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    z = np.zeros(data.shape[0])

    for i in xrange(n):
        c = np.random.randn(data.shape[0],1)
        c = c/np.sqrt((c**2).sum())
        cur_fps = []
        cur_edists = []
        cur_hdists = []

        for step in steps:
            kwargs = cth.get_solver_kwargs(data, step_size=step)
            kwargs['c'] = c
            soln = df_sv.fiber_solver(**kwargs)
            fp = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)
            cur_fps.append(fp)

            cur_edists.append(np.array([min(np.linalg.norm(data-fp[:,j].reshape(-1,1), axis=0)) for j in xrange(fp.shape[1])]))
            cur_hdists.append(np.array([min(utils.hdist(data[:,d],fp[:,j]) for d in xrange(data.shape[1])) for j in xrange(fp.shape[1])
                if not np.allclose(fp[:,j],z)]))

        fps.append(cur_fps)
        fps_edists.append(cur_edists)
        fps_hdists.append(cur_hdists)


    return fps,fps_edists,fps_hdists

def test_cont_time_c_equal_data(n, shape, c_probes=5, step_size=1e-2, tol1=0.1, tol2=1e-8, tol3=1, silent=True):
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    fps = []
    all_pm1 = []
    equal_data = []
    stable = []
    lin_combos = []
    match_hn_fps = []


    for i in xrange(n):
        if not silent:
            print('i={}'.format(i))

        data = gd.get_random_discrete(*shape)

        W = np.matmul(data, data.T)
        for j in range(W.shape[0]):
            W[j,j] = 0
        W *= (1.0/W.shape[0])

        cur_fps = []
        cur_all_pm1 = []
        cur_equal_data = []
        cur_stable = []
        cur_lin_combos = []
        cur_match_hn_fps = []

        cs = [data[:,j][:,np.newaxis] for j in xrange(shape[1])]
        for j in xrange(c_probes):
            cs.append(np.sum([np.random.random()*data[:,k] for k in xrange(shape[1])], axis=0)[:,np.newaxis])
        cs = [c/np.sqrt((c**2).sum()) for c in cs]

        for c in cs:
            kwargs = cth.get_solver_kwargs(data, step_size=step_size)
            kwargs['c'] = c

            soln = df_sv.fiber_solver(**kwargs)
            fp = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)

            cur_fps.append(fp)
            cap1 = []
            ced = []
            cstable = []
            clc = []
            cmhf = []
            for k in xrange(fp.shape[1]):
                if np.all(np.abs(np.abs(fp[:,k])-1)<tol1):
                    cap1.append(k)
                hdists = [utils.hdist(data[:,l], fp[:,k]) for l in xrange(shape[1])]
                if any(h==0 or h==100 for h in hdists):
                    ced.append(k)
                if np.all(0>np.real(np.linalg.eigvals(kwargs['Df'](fp[:,k][:,np.newaxis])))):
                    cstable.append(k)
                resids = np.linalg.lstsq(data, fp[:,k], rcond=None)[1]
                if len(resids)>0 and resids[0]<tol2:
                    clc.append(k)
                nonmatches = np.count_nonzero(np.matmul(W, fp[:,k])*fp[:,k] < 0)
                if nonmatches<tol3:
                    cmhf.append((k,nonmatches))

            cur_all_pm1.append(cap1)
            cur_equal_data.append(ced)
            cur_stable.append(cstable)
            cur_lin_combos.append(clc)
            cur_match_hn_fps.append(cmhf)

        fps.append(cur_fps)
        all_pm1.append(cur_all_pm1)
        equal_data.append(cur_equal_data)
        stable.append(cur_stable)
        lin_combos.append(cur_lin_combos)
        match_hn_fps.append(cur_match_hn_fps)

    return fps, all_pm1, equal_data, stable, lin_combos, match_hn_fps

def test_cont_time_v0_in_corners(n, shape, v_probes=5, step_size=1e-2, tol1=0.1, tol2=1e-8, silent=True):
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    fps = []
    all_pm1 = []
    equal_data = []
    stable = []
    lin_combos = []

    for i in xrange(n):
        if not silent:
            print('i={}'.format(i))
        
        data = gd.get_random_discrete(*shape)

        cur_fps = []
        cur_all_pm1 = []
        cur_equal_data = []
        cur_stable = []
        cur_lin_combos = []

        vs = [data[:,j][:,np.newaxis] for j in xrange(shape[1])]
        for j in xrange(v_probes):
            vs.append(np.random.choice([-1.0,1.1], size=(shape[0],))[:,np.newaxis])

        for v in vs:
            kwargs = cth.get_solver_kwargs(data, step_size=step_size)
            kwargs['v'] = v

            soln = df_sv.fiber_solver(**kwargs)
            fp = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)

            cur_fps.append(fp)
            cap1 = []
            ced = []
            cs = []
            clc = []
            for k in xrange(fp.shape[1]):
                if np.all(np.abs(np.abs(fp[:,k])-1)<tol1):
                    cap1.append(k)
                hdists = [utils.hdist(data[:,l], fp[:,k]) for l in xrange(shape[1])]
                if any(h==0 or h==100 for h in hdists):
                    ced.append(k)
                if np.all(0>np.real(np.linalg.eigvals(kwargs['Df'](fp[:,k][:,np.newaxis])))):
                    cs.append(k)
                if np.linalg.lstsq(data, fp[:,k], rcond=None)[1][0]<tol2:
                    clc.append(k)

            cur_all_pm1.append(cap1)
            cur_equal_data.append(ced)
            cur_stable.append(cs)
            cur_lin_combos.append(clc)

        fps.append(cur_fps)
        all_pm1.append(cur_all_pm1)
        equal_data.append(cur_equal_data)
        stable.append(cur_stable)
        lin_combos.append(cur_lin_combos)

    return fps, all_pm1, equal_data, stable, lin_combos

def test_standard_vs_cont_stability2(data, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-16):
    cont_fps = []

    hn = chc.Hopnet(data.shape[0])
    hn.learn(data)


    stable_matched_fps = np.zeros(len(gains), dtype=np.int_)
    unstable_matched_fps = np.zeros(len(gains), dtype=np.int_)
    stable_unmatched_fps = np.zeros(len(gains), dtype=np.int_)
    unstable_unmatched_fps = np.zeros(len(gains), dtype=np.int_)

    # Find fixed points of cont Hopnet at various gain values
    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        gains_ = [g/max_eval for g in gains]
    else:
        gains_ = gains


    for i,g in enumerate(gains_):
        if traversal:
            fps,_ = rf.run_solver(hn.W*g)

        else:
            fxV, _ = rfl.baseline_solver(hn.W*g)
            fps, _ = rfl.post_process_fxpts(hn.W*g, fxV)

        cont_fps.append(fps)

        sgn_fps = np.sign(fps)

        
        for j in xrange(sgn_fps.shape[1]):
            if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                if np.allclose(np.sign(np.dot(hn.W,sgn_fps[:,j])), sgn_fps[:,j], rtol=0, atol=tol): # What about 0?
                    stable_matched_fps[i] += 1
                else:
                    stable_unmatched_fps[i] += 1
            else:
                if np.allclose(np.sign(np.dot(hn.W,sgn_fps[:,j])), sgn_fps[:,j], rtol=0, atol=tol): # What about 0?
                    unstable_matched_fps[i] += 1
                else:
                    unstable_unmatched_fps[i] += 1

    return cont_fps,stable_matched_fps,unstable_matched_fps,stable_unmatched_fps,unstable_unmatched_fps

def test_s_vs_c_stability_wrapper2(args):
    shape, gains, dynamic, traversal, tol = args
    np.random.seed()
    data = gd.get_random_discrete(*shape)
    _,sm,um,su,uu = test_standard_vs_cont_stability2(data, gains, dynamic, traversal, tol)
    return sm,um,su,uu

def test_s_vs_c_stability_full2(n, shape, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-6):
    stable_matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    stable_unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)

    # for i in xrange(n):
    #     data = gd.get_random_discrete(*shape)
    #     _,_,sm,um,su,uu = test_standard_vs_cont_stability2(data, gains, dynamic, traversal, tol)
    #     stable_matched_fps[i] += np.sum(sm,axis=0)
    #     unstable_matched_fps[i] += np.sum(um,axis=0)
    #     stable_unmatched_fps[i] += su
    #     unstable_unmatched_fps[i] += uu
    pool = mp.Pool(16)
    res = pool.map(test_s_vs_c_stability_wrapper2, [(shape,gains,dynamic,traversal,tol)]*n, chunksize=2)

    for i in xrange(n):
        stable_matched_fps[i] += res[i][0]
        unstable_matched_fps[i] += res[i][1]
        stable_unmatched_fps[i] += res[i][2]
        unstable_unmatched_fps[i] += res[i][3]


    return stable_matched_fps,unstable_matched_fps,stable_unmatched_fps,unstable_unmatched_fps


def test_cont_time_c_various(n, shape, c_probes=5, tol1=0.01, tol2=1e-8, tol3=1, step_size=None, ef=None, silent=True):
    """
    Args:
        n:          Number of datasets to try
        shape:      Tuple describing shape of the data array
        c_probes:   Number of c's to sample for c generation methods involving sampling
        tol1:       Tolerance used to check whether a fixed point is +/-1
        tol2:       Tolerance used to check whether a fixed point is a linear combination of the data points
        tol3:       Tolerance used to check whether the sign of a fixed point matches a fixed point under the standard Hopfield model
        step_size:  Step size for the solver to use. None for dynamic
        ef:         Forward error for the solver to use. None for dynamic
        silent:     Flag to enable printing iteration number

    Return values:
        res_data:       Dictionary of results for selecting c as a (scaled) data point
        res_data_span:  Dictionary of results for selecting c in the span of the data
        res_svd:        Dictionary of results for selecting c using the SVD method
        res_W_kI:       Dictionary of results for selecting c using the W+kI method
    """

    res_data = {'fps': [], 'all_pm1': [], 'equal_data': [], 'stable': [], 'lin_combos': [], 'match_hn_fps': []}
    res_data_span = {'fps': [], 'all_pm1': [], 'equal_data': [], 'stable': [], 'lin_combos': [], 'match_hn_fps': []}
    res_svd = {'fps': [], 'all_pm1': [], 'equal_data': [], 'stable': [], 'lin_combos': [], 'match_hn_fps': []}
    res_W_kI = {'fps': [], 'all_pm1': [], 'equal_data': [], 'stable': [], 'lin_combos': [], 'match_hn_fps': []}

    pool = mp.Pool(16) # Max number of processes

    for i in xrange(n):
        if not silent:
            print('i={}'.format(i))

        data = gd.get_random_discrete(*shape)

        W = np.matmul(data, data.T)
        for j in range(W.shape[0]):
            W[j,j] = 0
        W *= (1.0/W.shape[0])


        # c at data points
        data_cs = [data[:,j][:,np.newaxis] for j in xrange(shape[1])]
        data_cs = [c/np.sqrt((c**2).sum()) for c in data_cs]

        # c in span of data points
        data_span_cs = []
        for j in xrange(c_probes):
            data_span_cs.append(np.sum([np.random.random()*data[:,k] for k in xrange(shape[1])], axis=0)[:,np.newaxis])
        data_span_cs = [c/np.sqrt((c**2).sum()) for c in data_span_cs]

        # c equal to lin combo from svd
        U,S,V = np.linalg.svd(W)
        svd_c = np.sum([U[:,j]*S[j] for j in xrange(U.shape[1])], axis=0)[:,np.newaxis]
        svd_c = svd_c/np.sqrt((svd_c**2).sum())
        svd_cs = [svd_c]

        # c in span of W + kI
        W_kI_cs = []
        W_kI = W + shape[1]*np.eye(W.shape[0])/float(shape[0])
        for j in xrange(c_probes):
            W_kI_cs.append(np.sum([np.random.random()*W_kI[:,k] for k in xrange(W_kI.shape[1])], axis=0)[:,np.newaxis])
        W_kI_cs = [c/np.sqrt((c**2).sum()) for c in W_kI_cs]


        out_data = pool.map_async(test_cont_time_c_proc, [(data, W, c, tol1, tol2, tol3, step_size, ef) for c in data_cs])
        out_data_span = pool.map_async(test_cont_time_c_proc, [(data, W, c, tol1, tol2, tol3, step_size, ef) for c in data_span_cs])
        out_svd = pool.map_async(test_cont_time_c_proc, [(data, W, c, tol1, tol2, tol3, step_size, ef) for c in svd_cs])
        out_W_kI = pool.map_async(test_cont_time_c_proc, [(data, W, c, tol1, tol2, tol3, step_size, ef) for c in W_kI_cs])


        # Wait for processes to be done
        out_data = out_data.get()
        out_data_span = out_data_span.get()
        out_svd = out_svd.get()
        out_W_kI = out_W_kI.get()


        res_data['fps'].append([r[0] for r in out_data])
        res_data['all_pm1'].append([r[1] for r in out_data])
        res_data['equal_data'].append([r[2] for r in out_data])
        res_data['stable'].append([r[3] for r in out_data])
        res_data['lin_combos'].append([r[4] for r in out_data])
        res_data['match_hn_fps'].append([r[5] for r in out_data])

        res_data_span['fps'].append([r[0] for r in out_data_span])
        res_data_span['all_pm1'].append([r[1] for r in out_data_span])
        res_data_span['equal_data'].append([r[2] for r in out_data_span])
        res_data_span['stable'].append([r[3] for r in out_data_span])
        res_data_span['lin_combos'].append([r[4] for r in out_data_span])
        res_data_span['match_hn_fps'].append([r[5] for r in out_data_span])

        res_svd['fps'].append([r[0] for r in out_svd])
        res_svd['all_pm1'].append([r[1] for r in out_svd])
        res_svd['equal_data'].append([r[2] for r in out_svd])
        res_svd['stable'].append([r[3] for r in out_svd])
        res_svd['lin_combos'].append([r[4] for r in out_svd])
        res_svd['match_hn_fps'].append([r[5] for r in out_svd])

        res_W_kI['fps'].append([r[0] for r in out_W_kI])
        res_W_kI['all_pm1'].append([r[1] for r in out_W_kI])
        res_W_kI['equal_data'].append([r[2] for r in out_W_kI])
        res_W_kI['stable'].append([r[3] for r in out_W_kI])
        res_W_kI['lin_combos'].append([r[4] for r in out_W_kI])
        res_W_kI['match_hn_fps'].append([r[5] for r in out_W_kI])

    return res_data, res_data_span, res_svd, res_W_kI


def test_cont_time_c_proc(args):
    try:
        data, W, c, tol1, tol2, tol3, step_size, ef = args

        duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

        kwargs = cth.get_solver_kwargs(data, step_size, ef)
        kwargs['c'] = c

        soln = df_sv.fiber_solver(**kwargs)
        fp = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)

        cap1 = []
        ced = []
        cstable = []
        clc = []
        cmhf = []
        for k in xrange(fp.shape[1]):
            if np.all(np.abs(np.abs(fp[:,k])-1)<tol1):
                cap1.append(k)
            hdists = [utils.hdist(data[:,l], fp[:,k]) for l in xrange(data.shape[1])]
            if any(h==0 or h==100 for h in hdists):
                ced.append(k)
            if np.all(0>np.real(np.linalg.eigvals(kwargs['Df'](fp[:,k][:,np.newaxis])))):
                cstable.append(k)
            resids = np.linalg.lstsq(data, fp[:,k], rcond=None)[1]
            if len(resids)>0 and resids[0]<tol2:
                clc.append(k)
            nonmatches = np.count_nonzero(np.matmul(W, fp[:,k])*fp[:,k] < 0)
            if nonmatches<tol3:
                cmhf.append((k,nonmatches))

        return fp, cap1, ced, cstable, clc, cmhf
    except Exception as e:
        trace = traceback.format_exc()
        import pickle
        f = open('trace_{}'.format(os.getpid()), 'wb')
        pickle.dump(trace, f)
        f.close()
        raise e


def test_standard_vs_cont_stability3(data, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-16):
    def arr_to_bit_string(arr):
        s = ''
        for e in np.sign(arr):
            if e > 0:
                s += '1'
            else:
                s += '0'
        return s

    cont_fps = []

    hn = chc.Hopnet(data.shape[0])
    hn.learn(data)


    stable_matched_fps = np.zeros(len(gains), dtype=np.int_)
    unstable_matched_fps = np.zeros(len(gains), dtype=np.int_)
    stable_unmatched_fps = np.zeros(len(gains), dtype=np.int_)
    unstable_unmatched_fps = np.zeros(len(gains), dtype=np.int_)
    stable_hn_fps_found = np.zeros(len(gains), dtype=np.int_)
    unstable_hn_fps_found = np.zeros(len(gains), dtype=np.int_)
    total_hn_fps_found = np.zeros(len(gains), dtype=np.int_)

    # Find fixed points of cont Hopnet at various gain values
    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        gains_ = [g/max_eval for g in gains]
    else:
        gains_ = gains


    for i,g in enumerate(gains_):
        hn.gain = g

        if traversal:
            fps,_ = rf.run_solver(hn.W*g)

        else:
            fxV, _ = rfl.baseline_solver(hn.W*g)
            fps, _ = rfl.post_process_fxpts(hn.W*g, fxV)

        cont_fps.append(fps)

        sgn_fps = np.sign(fps)

        cur_stable_hn_fps_found = set()
        cur_unstable_hn_fps_found = set()
        cur_total_hn_fps_found = set()

        for j in xrange(sgn_fps.shape[1]):
            if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                if np.allclose(np.sign(np.dot(hn.W,sgn_fps[:,j])), sgn_fps[:,j], rtol=0, atol=tol): # What about 0?
                    stable_matched_fps[i] += 1
                    cur_stable_hn_fps_found.add(arr_to_bit_string(sgn_fps[:,j]))
                    cur_total_hn_fps_found.add(arr_to_bit_string(sgn_fps[:,j]))
                else:
                    stable_unmatched_fps[i] += 1
            else:
                if np.allclose(np.sign(np.dot(hn.W,sgn_fps[:,j])), sgn_fps[:,j], rtol=0, atol=tol): # What about 0?
                    unstable_matched_fps[i] += 1
                    cur_unstable_hn_fps_found.add(arr_to_bit_string(sgn_fps[:,j]))
                    cur_total_hn_fps_found.add(arr_to_bit_string(sgn_fps[:,j]))
                else:
                    unstable_unmatched_fps[i] += 1

        stable_hn_fps_found[i] += len(cur_stable_hn_fps_found)
        unstable_hn_fps_found[i] += len(cur_unstable_hn_fps_found.difference(cur_stable_hn_fps_found))
        total_hn_fps_found[i] += len(cur_total_hn_fps_found)

    return cont_fps,stable_matched_fps,unstable_matched_fps,stable_unmatched_fps,unstable_unmatched_fps,stable_hn_fps_found,unstable_hn_fps_found,total_hn_fps_found

def test_s_vs_c_stability_wrapper3(args):
    shape, gains, dynamic, traversal, tol = args
    np.random.seed()
    data = gd.get_random_discrete(*shape)
    _,sm,um,su,uu,sfps,ufps,tfps = test_standard_vs_cont_stability3(data, gains, dynamic, traversal, tol)
    return sm,um,su,uu,sfps,ufps,tfps

def test_s_vs_c_stability_full3(n, shape, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-6):
    stable_matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_matched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    stable_unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_unmatched_fps = np.zeros((n,len(gains)), dtype=np.int_)
    stable_hn_fps_found = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_hn_fps_found = np.zeros((n,len(gains)), dtype=np.int_)
    total_hn_fps_found = np.zeros((n,len(gains)), dtype=np.int_)

    # for i in xrange(n):
    #     data = gd.get_random_discrete(*shape)
    #     _,_,sm,um,su,uu = test_standard_vs_cont_stability3(data, gains, dynamic, traversal, tol)
    #     stable_matched_fps[i] += np.sum(sm,axis=0)
    #     unstable_matched_fps[i] += np.sum(um,axis=0)
    #     stable_unmatched_fps[i] += su
    #     unstable_unmatched_fps[i] += uu
    pool = mp.Pool(16)
    res = pool.map(test_s_vs_c_stability_wrapper3, [(shape,gains,dynamic,traversal,tol)]*n, chunksize=2)

    for i in xrange(n):
        stable_matched_fps[i] += res[i][0]
        unstable_matched_fps[i] += res[i][1]
        stable_unmatched_fps[i] += res[i][2]
        unstable_unmatched_fps[i] += res[i][3]
        stable_hn_fps_found[i] += res[i][4]
        unstable_hn_fps_found[i] += res[i][5]
        total_hn_fps_found[i] += res[i][6]

    return stable_matched_fps,unstable_matched_fps,stable_unmatched_fps,unstable_unmatched_fps,stable_hn_fps_found,unstable_hn_fps_found,total_hn_fps_found


def test_cont_time_crit_pts(data, c=None, step_size=None, ef=None, low=-2, high=2, step=0.1, plot=True, tol=1e-4):
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    kwargs = cth.get_solver_kwargs(data, c=c, step_size=step_size, ef=ef, history=None)

    num = int((high-low)/step)+1
    coords = np.linspace(low, high, num=num)

    min_eval_lst = []
    coords_lst = []

    for i,x in enumerate(coords):
        for j,y in enumerate(coords):
            for k,z in enumerate(coords):
                min_eval_lst.append(np.amin(np.abs(np.linalg.eigvals(kwargs['Df'](np.array([[x],[y],[z]]))))))
                coords_lst.append((x,y,z))

    coords_lst = np.array(coords_lst)
    min_eval_lst = np.array(min_eval_lst)

    min_eval_fiber = np.array([coords_lst[i,:] for i in xrange(min_eval_lst.shape[0]) if min_eval_lst[i]<tol])

    soln = df_sv.fiber_solver(**kwargs)
    fps = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)

    V = np.concatenate(soln["Fiber trace"].points, axis=1)[:3,:]

    V = V[:, \
            (V[0,:] >= low) & \
            (V[0,:] <= high) & \
            (V[1,:] >= low) & \
            (V[1,:] <= high) & \
            (V[2,:] >= low) & \
            (V[2,:] <= high)]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(V[0,:],V[1,:],V[2,:],color='blue', linestyle='-', linewidth=1) # Fiber

        if len(min_eval_fiber.shape)>0:
            ax.plot(min_eval_fiber[:,0],min_eval_fiber[:,1],min_eval_fiber[:,2], color='red', linestyle='-', linewidth=1) # min eval fiber

        ax.scatter(fps[0,:],fps[1,:],fps[2,:], color='green') # fps

        ax.scatter(coords_lst[:,0],coords_lst[:,1],coords_lst[:,2], c=np.log(1e-32+np.array(min_eval_lst)))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    return min_eval_lst,soln

def test_cont_time_find_crit_pts_evo(n, data, c=None, step_size=None, ef=None, low=-2, high=2, step=0.1, plot=True, tol1=1e-6, tol2=1e-6, decay=0.8, iter_lim=20):

    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)

    kwargs = cth.get_solver_kwargs(data, c=c, step_size=step_size, ef=ef, history=None)

    crit_pts = []
    
    

    for i in xrange(n):
        probe = np.random.random((data.shape[0],1))*2-1
        eps = 0.1
        min_eval = 1
        ctr = 0
        while min_eval > tol1:
            cur_gen = [probe + (np.random.random((data.shape[0],1))*2-1)*eps for i in xrange(9)]
            cur_gen.append(probe)
            for p in cur_gen:
                e = np.amin(np.abs(np.linalg.eigvals(kwargs['Df'](p))))
                if e < min_eval:
                    min_eval = e
                    probe = p
            eps *= decay
            if ctr == iter_lim:
                break
            ctr += 1
        else:
            if n<1000:
                if not np.any([np.allclose(probe, cp, rtol=0, atol=tol2) for cp in crit_pts]):
                    crit_pts.append(probe)
            else:
                crit_pts.append(probe)



    return np.array(crit_pts), kwargs


def test_tanh_capacity_n_vs_d_process(args):
    tests,ns,d,gain,sync = args
    np.random.seed()

    if sync:
        mode = 'sync'
    else:
        mode = 'async_deterministic'
    if gain == 'sqrt(n)':
        gain = np.sqrt(n)

    errs = np.zeros((len(ns),tests))

    for i,n in enumerate(ns):
        for j in xrange(tests):
            data = gd.get_random_discrete(n,d)
            hn = chc.Hopnet(n, mode=chc.modes[mode], gain=gain)
            hn.learn(data)

            act = hn.simhop(data[:,0], silent=True)[1] # Arbitrarily pick the first data point to avoid scaling issues when changing d
            errs[i,j] = utils.hdist(act,data[:,0])/float(n)

    return errs

def test_tanh_capacity_n_vs_d(tests, ns, ds, gain, sync=False):
    pool = mp.Pool(16)

    errs = np.array(pool.map(test_tanh_capacity_n_vs_d_process, [(tests,ns,d,gain,sync) for d in ds])) # Shape: ds*ns*tests

    return np.transpose(errs, (1,0,2)) # Shape: ns*ds*tests


def test_stability_at_mems2(n=32, size=100, gain=10, num_mems=[1,5,10,15,20], num_bits_to_flip=[0,5,10,15,20,25,30,35,40,45,50], probes=50):
    
    pool = mp.Pool(16)

    pbfs = pool.map(test_stability_at_mems2_process, [(size,gain,num_mems,num_bits_to_flip,probes)]*n)

    return np.array(pbfs)


def test_stability_at_mems2_process(args):
    size,gain,num_mems,num_bits_to_flip,probes = args

    pbfs = np.zeros((len(num_mems),len(num_bits_to_flip)))

    hn = chc.Hopnet(size, gain=gain)

    for j,nm in enumerate(num_mems):
        for k,nbtf in enumerate(num_bits_to_flip):
            data = gd.get_random_discrete(size,nm)
            hn.learn(data)

            for mem in xrange(nm):
                for l in xrange(probes):
                    probe = np.copy(data[:,mem])
                    flipped_bits = np.random.choice(xrange(size), size=nbtf, replace=False)
                    for bit in flipped_bits:
                        probe[bit] *= -1
                    hn.simhop(probe, silent=True)
                    pbfs[j,k] += utils.hdist(data[:,mem], hn.a) # maybe hdist_z?
            pbfs[j,k] /= float(size*nm*probes)

    return pbfs



def test_stability_at_fp_4(hn, fp, probes=100):
    percent_bits_flipped = []


    for i in xrange(1+hn.n/2):
        cur_bf = []

        for j in xrange(probes):
            probe = np.copy(fp)
            flipped_bits = np.random.choice(xrange(hn.n), size=i, replace=False)
            for bit in flipped_bits:
                probe[bit] *= -1
            hn.simhop(probe, silent=True)
            cur_bf.append(utils.hdist(fp, hn.a)/float(hn.n)) # maybe hdist_z?

        percent_bits_flipped.append(cur_bf)

    return np.array(percent_bits_flipped)

def test_stability_4_process(W):
    np.random.seed()
    return rf.run_solver(W)

def test_stability_4(size=(100,3), data=None, gain=10, traversal_attempts=10, probes=100):

    pbfs = []
    jacobians = []

    if data is not None:
        size = data.shape
    else:
        data = gd.get_random_discrete(*size)

    hn = chc.Hopnet(size[0], gain=gain)
    hn.learn(data)


    pool = mp.Pool(16)

    fps_lst = pool.imap(test_stability_4_process, (hn.W*gain for _ in xrange(traversal_attempts)))
    fps_lst = [fp for fp,_ in fps_lst]
    fps = np.concatenate(fps_lst, axis=1)
    fps,_ = rf.post_process_fxpts(hn.W*gain, fps)

    # fps,_ = rf.run_solver(hn.W*gain)
    # for i in xrange(traversal_attempts-1):
    #     new_fps,_ = rf.run_solver(hn.W*gain)
    #     fps = np.concatenate((fps, new_fps), axis=1)
    #     fps,_ = rf.post_process_fxpts(hn.W*hn.gain, fps)


    for i in xrange(fps.shape[1]):
        pbf = test_stability_at_fp_4(hn, fps[:,i], probes=probes)
        pbfs.append(pbf)
        jacobians.append(np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,i])))))

    return data, fps, np.array(pbfs), jacobians


def test_fps_vs_runs_tanh_process(args):
    np.random.seed()

    size,gain,traversal_attempts = args

    stable_tanh_fps = [0 for _ in xrange(traversal_attempts)]
    unstable_tanh_fps = [0 for _ in xrange(traversal_attempts)]
    hn_fps = [0 for _ in xrange(traversal_attempts)]

    data = gd.get_random_discrete(*size)

    hn = chc.Hopnet(size[0], gain=gain)
    hn.learn(data)

    fps = None

    for i in xrange(traversal_attempts):
        new_fps,_ = rf.run_solver(hn.W*gain)
        if fps is None:
            fps = new_fps
        else:
            fps = np.concatenate((fps,new_fps), axis=1)
            fps,_ = rf.post_process_fxpts(hn.W*gain, fps)

        cur_hn_fps = []
        for j in xrange(fps.shape[1]):
            if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                stable_tanh_fps[i] += 1
                sgn_cur_fp = np.sign(fps[:,j])
                for fp in cur_hn_fps:
                    if np.allclose(sgn_cur_fp, fp, rtol=0, atol=1e-6):
                        break
                else:
                    hn_fps[i] += 1
                    cur_hn_fps.append(sgn_cur_fp)

            else:
                unstable_tanh_fps[i] += 1

    return stable_tanh_fps, unstable_tanh_fps, hn_fps

def test_fps_vs_runs_tanh(n=10, size=(100,3), gain=10, traversal_attempts=10):
    stable_tanh_fps = []
    unstable_tanh_fps = []
    hn_fps = []

    pool = mp.Pool(16)

    fps_lst = pool.map(test_fps_vs_runs_tanh_process, [(size,gain,traversal_attempts) for _ in xrange(n)])

    for s,u,hn in fps_lst:
        stable_tanh_fps.append(s)
        unstable_tanh_fps.append(u)
        hn_fps.append(hn)

    return np.array(stable_tanh_fps), np.array(unstable_tanh_fps), np.array(hn_fps)


def test_stability_at_mems3(n=50, size=100, gain=10, num_mems=[1,5,10,15,20], num_bits_to_flip=[0,5,10,15,20,25,30,35,40,45,50], 
    ini_mults=[0.1,0.5,0.9], probes=50):
    """
    test_stability_at_mems2 rewritten for quadratic model.
    Basically redoing test_jr_ode_content_addr
    """

    pool = mp.Pool(16)

    pbfs = pool.map(test_stability_at_mems3_process, [(size,gain,num_mems,num_bits_to_flip,ini_mults,probes)]*n)

    return np.array(pbfs)


def test_stability_at_mems3_process(args):
    np.random.seed()
    
    def dvdt(v, t, W):
        return np.dot(W,v)*(1-v**2)

    size,gain,num_mems,num_bits_to_flip,ini_mults,probes = args

    time = np.linspace(0, 50, 50*10)

    pbfs = np.zeros((len(ini_mults), len(num_mems),len(num_bits_to_flip)))

    hn = chc.Hopnet(size, gain=gain)

    for i,im in enumerate(ini_mults):
        for j,nm in enumerate(num_mems):
            for k,nbtf in enumerate(num_bits_to_flip):
                data = gd.get_random_discrete(size,nm)
                hn.learn(data)

                for mem in xrange(nm):
                    for l in xrange(probes):
                        probe = np.copy(data[:,mem])
                        flipped_bits = np.random.choice(xrange(size), size=nbtf, replace=False)
                        for bit in flipped_bits:
                            probe[bit] *= -1
                        ode_res = spi.odeint(dvdt, probe*im, time, args=(hn.W,))
                        res_jr = ode_res[-1,:]
                        pbfs[i,j,k] += utils.hdist(data[:,mem], res_jr) # maybe hdist_z?
                pbfs[i,j,k] /= float(size*nm*probes)
    return pbfs


def test_standard_vs_cont_stability4(data, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-16):
    def arr_to_bit_string(arr):
        s = ''
        for e in np.sign(arr):
            if e > 0:
                s += '1'
            else:
                s += '0'
        return s

    cont_fps = []

    hn = chc.Hopnet(data.shape[0])
    hn.learn(data)


    stable_match_data = np.zeros(len(gains), dtype=np.int_)
    unstable_match_data = np.zeros(len(gains), dtype=np.int_)
    stable_nmatch_data = np.zeros(len(gains), dtype=np.int_)
    unstable_nmatch_data = np.zeros(len(gains), dtype=np.int_)
    stable_match_hn_fps = np.zeros(len(gains), dtype=np.int_)
    unstable_match_hn_fps = np.zeros(len(gains), dtype=np.int_)
    stable_nmatch_hn_fps = np.zeros(len(gains), dtype=np.int_)
    unstable_nmatch_hn_fps = np.zeros(len(gains), dtype=np.int_)
    num_matched_data_by_all = np.zeros(len(gains), dtype=np.int_)
    num_matched_data_by_stable = np.zeros(len(gains), dtype=np.int_)
    num_matched_data_by_unstable = np.zeros(len(gains), dtype=np.int_)
    num_matched_hn_fps_by_all = np.zeros(len(gains), dtype=np.int_)
    num_matched_hn_fps_by_stable = np.zeros(len(gains), dtype=np.int_)
    num_matched_hn_fps_by_unstable = np.zeros(len(gains), dtype=np.int_)

    timings = np.zeros(len(gains), dtype=np.int_)

    # Find fixed points of cont Hopnet at various gain values
    if dynamic:
        max_eval = np.max(np.linalg.eigvalsh(hn.W))
        gains_ = [g/max_eval for g in gains]
    else:
        gains_ = gains


    for i,g in enumerate(gains_):
        hn.gain = g

        if traversal:
            pre = utils.process_time()
            fps, _ = rftpp.run_solver(hn.W*g)
            post = utils.process_time()

        else:
            pre = utils.process_time()
            fxV, _ = rftpp.baseline_solver(hn.W*g)
            fps, _ = rftpp.post_process_fxpts(hn.W*g, fxV, neighbors=lambda X,y: (np.fabs(X-y)<2**-21).all(axis=0))
            post = utils.process_time()

        timings[i] = post-pre

        cont_fps.append(fps)

        sgn_fps = np.sign(fps)

        cur_matched_data_by_all = set()
        cur_matched_data_by_stable = set()
        cur_matched_data_by_unstable = set()
        cur_matched_hn_fps_by_all = set()
        cur_matched_hn_fps_by_stable = set()
        cur_matched_hn_fps_by_unstable = set()

        for j in xrange(sgn_fps.shape[1]):
            fp_str = arr_to_bit_string(sgn_fps[:,j])

            if np.all(1>np.absolute(np.linalg.eigvals(hn.jacobian(fps[:,j])))):
                if np.allclose(np.sign(np.dot(hn.W,sgn_fps[:,j])), sgn_fps[:,j], rtol=0, atol=tol): # What about 0?
                    stable_match_hn_fps[i] += 1

                    cur_matched_hn_fps_by_all.add(fp_str)
                    cur_matched_hn_fps_by_stable.add(fp_str)
                else:
                    stable_nmatch_hn_fps[i] += 1

                if any(np.allclose(np.sign(data[:,k]), sgn_fps[:,j], rtol=0, atol=tol) for k in xrange(data.shape[1])):
                    stable_match_data[i] += 1

                    cur_matched_data_by_all.add(fp_str)
                    cur_matched_data_by_stable.add(fp_str)
                else:
                    stable_nmatch_data[i] += 1                    

            else:
                if np.allclose(np.sign(np.dot(hn.W,sgn_fps[:,j])), sgn_fps[:,j], rtol=0, atol=tol): # What about 0?
                    unstable_match_hn_fps[i] += 1

                    cur_matched_hn_fps_by_all.add(fp_str)
                    cur_matched_hn_fps_by_unstable.add(fp_str)
                else:
                    unstable_nmatch_hn_fps[i] += 1

                if any(np.allclose(np.sign(data[:,k]), sgn_fps[:,j], rtol=0, atol=tol) for k in xrange(data.shape[1])):
                    unstable_match_data[i] += 1

                    cur_matched_data_by_all.add(fp_str)
                    cur_matched_data_by_unstable.add(fp_str)
                else:
                    unstable_nmatch_data[i] += 1     

        num_matched_data_by_all[i] = len(cur_matched_data_by_all)
        num_matched_data_by_stable[i] = len(cur_matched_data_by_stable)
        num_matched_data_by_unstable[i] = len(cur_matched_data_by_unstable)
        num_matched_hn_fps_by_all[i] = len(cur_matched_hn_fps_by_all)
        num_matched_hn_fps_by_stable[i] = len(cur_matched_hn_fps_by_stable)
        num_matched_hn_fps_by_unstable[i] = len(cur_matched_hn_fps_by_unstable)
        
    res = (stable_match_data, unstable_match_data, stable_nmatch_data, unstable_nmatch_data,
            stable_match_hn_fps, unstable_match_hn_fps, stable_nmatch_hn_fps, unstable_nmatch_hn_fps,
            num_matched_data_by_all, num_matched_data_by_stable, num_matched_data_by_unstable,
            num_matched_hn_fps_by_all, num_matched_hn_fps_by_stable, num_matched_hn_fps_by_unstable,
            timings)
    return res

def test_s_vs_c_stability_wrapper4(args):
    shape, gains, dynamic, traversal, tol = args
    np.random.seed()
    data = gd.get_random_discrete(*shape)
    res = test_standard_vs_cont_stability4(data, gains, dynamic, traversal, tol)
    return res

def test_s_vs_c_stability_full4(n, shape, gains=[1.0,10.0,50.0,100.0], dynamic=False, traversal=True, tol=1e-6):
    stable_match_data = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_match_data = np.zeros((n,len(gains)), dtype=np.int_)
    stable_nmatch_data = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_nmatch_data = np.zeros((n,len(gains)), dtype=np.int_)
    stable_match_hn_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_match_hn_fps = np.zeros((n,len(gains)), dtype=np.int_)
    stable_nmatch_hn_fps = np.zeros((n,len(gains)), dtype=np.int_)
    unstable_nmatch_hn_fps = np.zeros((n,len(gains)), dtype=np.int_)
    num_matched_data_by_all = np.zeros((n,len(gains)), dtype=np.int_)
    num_matched_data_by_stable = np.zeros((n,len(gains)), dtype=np.int_)
    num_matched_data_by_unstable = np.zeros((n,len(gains)), dtype=np.int_)
    num_matched_hn_fps_by_all = np.zeros((n,len(gains)), dtype=np.int_)
    num_matched_hn_fps_by_stable = np.zeros((n,len(gains)), dtype=np.int_)
    num_matched_hn_fps_by_unstable = np.zeros((n,len(gains)), dtype=np.int_)
    timings = np.zeros((n,len(gains)), dtype=np.int_)


    pool = mp.Pool(16)
    res = pool.map(test_s_vs_c_stability_wrapper4, [(shape,gains,dynamic,traversal,tol)]*n, chunksize=2)

    for i in xrange(n):
        stable_match_data[i] += res[i][0]
        unstable_match_data[i] += res[i][1]
        stable_nmatch_data[i] += res[i][2]
        unstable_nmatch_data[i] += res[i][3]
        stable_match_hn_fps[i] += res[i][4]
        unstable_match_hn_fps[i] += res[i][5]
        stable_nmatch_hn_fps[i] += res[i][6]
        unstable_nmatch_hn_fps[i] += res[i][7]
        num_matched_data_by_all[i] += res[i][8]
        num_matched_data_by_stable[i] += res[i][9]
        num_matched_data_by_unstable[i] += res[i][10]
        num_matched_hn_fps_by_all[i] += res[i][11]
        num_matched_hn_fps_by_stable[i] += res[i][12]
        num_matched_hn_fps_by_unstable[i] += res[i][13]
        timings[i] += res[i][14]

    res = (stable_match_data, unstable_match_data, stable_nmatch_data, unstable_nmatch_data,
            stable_match_hn_fps, unstable_match_hn_fps, stable_nmatch_hn_fps, unstable_nmatch_hn_fps,
            num_matched_data_by_all, num_matched_data_by_stable, num_matched_data_by_unstable,
            num_matched_hn_fps_by_all, num_matched_hn_fps_by_stable, num_matched_hn_fps_by_unstable,
            timings)
    return res

