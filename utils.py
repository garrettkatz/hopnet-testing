"""
Various utilities. hdist is the only one I really use right now.
"""

import numpy as np
import time

def hdist(a,b):
    return np.sum((a>0) != (b>0))

def hdist_z(a,b, tol=1e-6):
    a_pos = a>tol
    a_neg = a<-tol
    a_zero = np.abs(a)<tol

    b_pos = b>tol
    b_neg = b<-tol
    b_zero = np.abs(b)<tol

    return np.count_nonzero(np.logical_and(a_pos,b_neg)) \
        + np.count_nonzero(np.logical_and(a_neg,b_pos)) \
        + 0.5*np.count_nonzero(np.logical_xor(a_zero,b_zero))

def match_sets(set_lst):
    """
    set_lst is of type matrix[d*n] list; that is, it is a list of matrices where 
    the [:,i]th element is a vector (typically a fixed point).
    This follows the format of fixed point sets in rnn_fxpts.
    It is assumed that each matrix contains a set of distinct vectors.
    This function returns a list of type (vector[d] * int list) list
    Each vector is a fixed point; the int list counts which matrices the vector is in.
    """

    consolidated = []

    for i,s in enumerate(set_lst):
        for j in xrange(s.shape[1]):
            for v,counts in consolidated:
                if hdist_z(v,s[:,j]) == 0:
                    counts.append(i)
                    break
            else:
                consolidated.append((s[:,j],[i]))

    return consolidated

def jdist(a,b):
    """
    Jaccard distance between two matrices of fixed points in the format
    returned in rnn_fxpts.
    Jaccard distance between two sets is 
    1 - |A intersect B|/|A union B| = 1 - |A intersect B|/(|A|+|B|-|A intersect B|)
    """

    consolidated = match_sets([a,b])

    intersection = 0
    for _,counts in consolidated:
        if counts == [0,1]:
            intersection += 1
    return 1 - float(intersection)/(a.shape[1] + b.shape[1] - intersection)


def wait_until_cool(max_temp):
    """
    Wait until processors cool down below a given temperature threshold (in C)
    """
    f = open('/sys/class/thermal/thermal_zone0/temp','r')
    while int(f.read()) > max_temp*1000:
        # Need to close and reopen file because it will be updated
        f.close()
        time.sleep(0.1)
        f = open('/sys/class/thermal/thermal_zone0/temp','r')
    f.close()