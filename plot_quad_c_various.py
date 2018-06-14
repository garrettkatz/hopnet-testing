"""
Experiment with visualizing the fibers from the quadratic model using PCA.
Doesn't work particularly well.
"""

import multiprocessing as mp
import numpy as np
from sklearn.decomposition import PCA
import gen_data as gd
import cont_time_hopnet as cth
import dfibers.fixed_points as df_fx
import dfibers.solvers as df_sv
import utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def gen_trace(kwargs):
    duplicates = lambda U, v: (np.fabs(U - v) < 1e-6).all(axis=0)
    soln = df_sv.fiber_solver(**kwargs)
    fps = df_fx.sanitize_points(soln['Fixed points'], kwargs['f'], kwargs['ef'], kwargs['Df'], duplicates)

    return fps, soln['Fiber trace'].points, soln['Fiber trace'].step_amounts

def plot(data, c, fps, points, step_amounts):
    V = np.concatenate(points, axis=1)[:-1,:-1] #-1 in first axis removes extra dimension, in second removes last point which isn't in step_amounts

    pca = PCA(n_components=3)
    pca.fit(V.T)

    V_proj = np.array([np.matmul(pca.components_, V[:,i]) for i in xrange(V.shape[1])])
    data_proj = np.array([np.matmul(pca.components_, data[:,i]) for i in xrange(data.shape[1])])
    c_proj = np.matmul(pca.components_, c).flatten()
    fps_proj = np.array([np.matmul(pca.components_, fps[:,i]) for i in xrange(fps.shape[1])])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(V_proj[:,0], V_proj[:,1], V_proj[:,2], c=np.log(step_amounts), marker='.')

    scale = 0.25*min(np.max(np.abs(V_proj[:,0])), np.max(np.abs(V_proj[:,1])), np.max(np.abs(V_proj[:,2])))
    for i in xrange(data_proj.shape[0]):
        d_line = np.stack((scale*data_proj[i], -scale*data_proj[i]))/np.linalg.norm(data_proj[i])
        ax.plot(d_line[:,0], d_line[:,1], d_line[:,2], c='blue')

    c_line = np.stack((scale*c_proj, -scale*c_proj))
    ax.plot(c_line[:,0], c_line[:,1], c_line[:,2], c='red')

    ax.scatter(fps_proj[:,0], fps_proj[:,1], fps_proj[:,2], c='green', marker='D')

    plt.show()

def all_plot(data, c_lst, fps_lst, points_lst, eval_fps):
    colors = ['red', 'green', 'blue']
    labels = ['data span', 'svd', 'W+kI']
    scale = 0.5

    V_lst = [np.concatenate(points, axis=1)[:-1,:] for points in points_lst]
    V_all = np.concatenate(V_lst, axis=1)

    pca = PCA(n_components=3)
    pca.fit(V_all.T)

    V_proj_lst = [np.array([np.matmul(pca.components_, V[:,i]) for i in xrange(V.shape[1])]) for V in V_lst]
    data_proj = np.array([np.matmul(pca.components_, data[:,i]) for i in xrange(data.shape[1])])
    c_proj_lst = [np.matmul(pca.components_, c).flatten() for c in c_lst]
    fps_proj_lst = [np.array([np.matmul(pca.components_, fps[:,i]) for i in xrange(fps.shape[1])]) for fps in fps_lst]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in xrange(len(c_lst)):
        ax.scatter(V_proj_lst[i][:,0], V_proj_lst[i][:,1], V_proj_lst[i][:,2], c=colors[i], marker='.', label=labels[i])

        c_line = np.stack((scale*c_proj_lst[i], -scale*c_proj_lst[i]))
        ax.plot(c_line[:,0], c_line[:,1], c_line[:,2], c='dark'+colors[i])

        ax.scatter(fps_proj_lst[i][:,0], fps_proj_lst[i][:,1], fps_proj_lst[i][:,2], 
            c=['magenta' if eval_fps(fps_lst[i][:,j],c_lst[i]) else 'orange' for j in xrange(fps_lst[i].shape[1])], marker='D')

        
    for i in xrange(data_proj.shape[0]):
        d_line = np.stack((scale*data_proj[i], -scale*data_proj[i]))/np.linalg.norm(data_proj[i])
        ax.plot(d_line[:,0], d_line[:,1], d_line[:,2], c='black')

    plt.legend()

    plt.show()

def plot_compare_runs(data, dyn, const, eval_fps):
    def unitize(x):
        return x/np.linalg.norm(x)

    c_lst_d, fps_lst_d, points_lst_d = dyn
    c_lst_c, fps_lst_c, points_lst_c = const

    c_lst = c_lst_d + c_lst_c
    points_lst = points_lst_d + points_lst_c
    fps_lst = fps_lst_d + fps_lst_c

    colors = ['red', 'green', 'blue', 'cyan', 'salmon', 'violet']
    labels = ['Dynamic: data span', 'Dynamic: svd', 'Dynamic: W+kI', 'Constant: data span', 'Constant: svd', 'Constant: W+kI']
    scale = 0.5

    V_lst = [np.concatenate(points, axis=1)[:-1,:] for points in points_lst]
    V_all = np.concatenate(V_lst, axis=1)

    pca = PCA(n_components=3)
    pca.fit(V_all.T)
    # fps_all = np.concatenate(fps_lst, axis=1)
    # pca.fit(fps_all.T)
    # c_all = np.concatenate(c_lst, axis=1)
    # pca.fit(c_all.T)


    V_proj_lst = [np.array([unitize(np.matmul(pca.components_, V[:,i]))*np.linalg.norm(V[:,i]) for i in xrange(V.shape[1])]) for V in V_lst]
    data_proj = np.array([unitize(np.matmul(pca.components_, data[:,i]))*np.linalg.norm(data[:,i]) for i in xrange(data.shape[1])])
    c_proj_lst = [unitize(np.matmul(pca.components_, c.flatten()))*np.linalg.norm(c.flatten()) for c in c_lst]
    fps_proj_lst = [np.array([unitize(np.matmul(pca.components_, fps[:,i]))*np.linalg.norm(fps[:,i]) for i in xrange(fps.shape[1])]) for fps in fps_lst]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in xrange(len(c_lst)):
        ax.scatter(V_proj_lst[i][:,0], V_proj_lst[i][:,1], V_proj_lst[i][:,2], c=colors[i], marker='.', label=labels[i])

        c_line = np.stack((scale*c_proj_lst[i], -scale*c_proj_lst[i]))
        ax.plot(c_line[:,0], c_line[:,1], c_line[:,2], c='dark'+colors[i])

        ax.scatter(fps_proj_lst[i][:,0], fps_proj_lst[i][:,1], fps_proj_lst[i][:,2], 
            c=['magenta' if eval_fps(fps_lst[i][:,j],c_lst[i]) else 'orange' for j in xrange(fps_lst[i].shape[1])], marker='D')

        
    for i in xrange(data_proj.shape[0]):
        d_line = np.stack((scale*data_proj[i], -scale*data_proj[i]))/np.linalg.norm(data_proj[i])
        ax.plot(d_line[:,0], d_line[:,1], d_line[:,2], c='black')

    plt.legend()

    plt.show()


def eval_all_pm1(data):
    def f(fp,_):
        return np.all(np.abs(np.abs(fp)-1)<0.01)
    return f

def eval_equal_data(data):
    def f(fp,_):
        hdists = [utils.hdist(data[:,l], fp) for l in xrange(data.shape[1])]
        return any(h==0 or h==100 for h in hdists)
    return f

def eval_stable(data):
    def f(fp,c):
        kwargs = cth.get_solver_kwargs(data, c=c)
        np.all(0>np.real(np.linalg.eigvals(kwargs['Df'](fp[:,np.newaxis]))))
    return f

def eval_match_hn_fps(data):
    W = np.matmul(data, data.T)
    for i in range(W.shape[0]):
        W[i,i] = 0
    W *= (1.0/W.shape[0])
    def f(fp,c):
        return np.count_nonzero(np.matmul(W, fp)*fp < 0) == 0
    return f




def random_run(disp=True):
    data = gd.get_random_discrete(100,3)

    c = np.random.randn(100,1)
    c = c/np.sqrt((c**2).sum())

    kwargs = cth.get_solver_kwargs(data, c=c, step_size=None, ef=None, history=None)

    fps, points, step_amounts = gen_trace(kwargs)

    if disp:
        plot(data, c, fps, points, step_amounts)

    return data, c, fps, points, step_amounts

def all_run_proc(args):
    data, c, step_size, ef = args

    kwargs = cth.get_solver_kwargs(data, c=c, step_size=step_size, ef=ef, history=None)
    return gen_trace(kwargs)


def all_run(step_size, ef, shape=(100,3), disp=True, data=None):

    if data is not None:
        data = gd.get_random_discrete(*shape)

    W = np.matmul(data, data.T)
    for j in range(W.shape[0]):
        W[j,j] = 0
    W *= (1.0/W.shape[0])

    # # c at data points
    # data_cs = [data[:,j][:,np.newaxis] for j in xrange(shape[1])]
    # data_cs = [c/np.sqrt((c**2).sum()) for c in data_cs]

    # c in span of data points
    data_span_c = np.sum([np.random.random()*data[:,k] for k in xrange(shape[1])], axis=0)[:,np.newaxis]
    data_span_c = data_span_c/np.sqrt((data_span_c**2).sum())

    # c equal to lin combo from svd
    U,S,V = np.linalg.svd(W)
    svd_c = np.sum([U[:,j]*S[j] for j in xrange(U.shape[1])], axis=0)[:,np.newaxis]
    svd_c = svd_c/np.sqrt((svd_c**2).sum())

    # c in span of W + kI
    W_kI = W + shape[1]*np.eye(W.shape[0])/float(shape[0])
    W_kI_c = np.sum([np.random.random()*W_kI[:,k] for k in xrange(W_kI.shape[1])], axis=0)[:,np.newaxis]
    W_kI_c = W_kI_c/np.sqrt((W_kI_c**2).sum())

    c_lst = [data_span_c, svd_c, W_kI_c]

    pool = mp.Pool()

    res_lst = pool.map(all_run_proc, [(data, data_span_c, step_size, ef), (data, svd_c, step_size, ef), (data, W_kI_c, step_size, ef)])
    fps_lst = []
    points_lst = []
    steps_lst = []
    for fps,points,steps in res_lst:
        fps_lst.append(fps)
        points_lst.append(points[::100])
        steps_lst.append(steps[::100])

    if disp:
        all_plot(data, c_lst, fps_lst, points_lst)

    return data, c_lst, fps_lst, points_lst, steps_lst

