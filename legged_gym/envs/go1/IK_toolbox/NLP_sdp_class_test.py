# -*- encoding: UTF-8 -*-
########test
import numpy as np
import math
import string

import copy
import sys
import mosek

import time
from functools import wraps

from KMP_class import KMP

from NLP_sdp_class import NLP


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.func_name, str(t1 - t0))
              )
        return result

    return function_timer


# load global variables from external files: not using here
##import global_variables


def main():
    nstep = 15
    dt_sample = 0.005
    dt_nlp = 0.05
    hcomx = 0.466
    sx = 0.1
    sy = 0.145
    sz = 0
    st = 0.799

    ### KMP initiallization
    rleg_traj_refx = np.loadtxt(
        "/home/jiatao/Documents/cvx-a64/cvx/examples/NMPC_QCQP_solution/numerical optimization_imitation learning/nlp_nao_experiments/referdata_swing_mod.txt")  ### linux/ubuntu
    lleg_traj_refx = np.loadtxt(
        "/home/jiatao/Documents/cvx-a64/cvx/examples/NMPC_QCQP_solution/numerical optimization_imitation learning/nlp_nao_experiments/referdata_swing_mod.txt")  ### linux/ubuntu

    # rleg_traj_refx = np.loadtxt(r'D:\research\numerical optimization_imitation learning\nlp_nao_experiments\referdata_swing_mod.txt') ###window
    # lleg_traj_refx = np.loadtxt(r'D:\research\numerical optimization_imitation learning\nlp_nao_experiments\referdata_swing_mod.txt')  ###window


    inDim = 1  ### time as input
    outDim = 6  ### decided by traj_Dim * (pos+?vel+?acc: indicated by pvFlag)
    kh = 2
    lamda = 1
    pvFlag = 1  ## pvFlag = 0(pos) & pvFlag = 1 (pos+vel)


    com_fo_nlp =  NLP(nstep,dt_sample,dt_nlp,hcomx, sx,sy,sz,st,rleg_traj_refx, lleg_traj_refx, inDim, outDim, kh, lamda, pvFlag)
    print(com_fo_nlp.footy_ref)
    print(com_fo_nlp.Nsum)
    print(com_fo_nlp.Nsum1)
    outx = np.zeros([com_fo_nlp.Nsum, 12])
    RLfoot_com_pos = np.zeros([9, com_fo_nlp.Nsum1])

    t_n = int(2 * np.round(com_fo_nlp.Ts[0, 0] / dt_sample))


    for i in range(1, com_fo_nlp.Nsum1 - t_n):
        # for i in range(1, 5):
        time1 = time.time()
        j_index = int(np.floor((i) / (com_fo_nlp.dt / com_fo_nlp.dtx)))  ####walking time fall into a specific optmization loop
        if ((j_index >= 1) and (abs(i * com_fo_nlp.dtx - j_index * com_fo_nlp.dt) <= 0.8 * dt_sample)):
            res_outx = com_fo_nlp.nlp_nao(j_index)
            outx[j_index - 1, :] = res_outx

        rfoot_p, lfoot_p = com_fo_nlp.kmp_foot_trajectory(i, dt_sample, j_index, rleg_traj_refx, lleg_traj_refx, inDim, outDim, kh, lamda, pvFlag)
        RLfoot_com_pos[0:3, i - 1] = copy.deepcopy(rfoot_p[0:3, 0])
        RLfoot_com_pos[3:6, i - 1] = copy.deepcopy(lfoot_p[0:3, 0])

        com_intex = com_fo_nlp.XGetSolution_CoM_position(i, dt_sample, j_index)
        RLfoot_com_pos[6:9, i - 1] = copy.deepcopy(com_intex[0:3, 0])

        time2 = time.time()
        if ((time2 - time1) >= 0.015):
            print(" the time of (%d) kmp is  %3f" % (i, time2 - time1))
        # print('j_index is (%d)' %j_index)
        # print(i*dtx - j_index*dt)

    # ## save data in a txt file
    np.savetxt(
        "/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/pybullet_robots/locomotion_nmpc_pybullet_python/py_nlp_sdp_com.txt",
        outx, fmt="%f")
    np.savetxt(
        "/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/pybullet_robots/locomotion_nmpc_pybullet_python/py_kmp_foot_traj.txt",
        RLfoot_com_pos, fmt="%f")
    # np.savetxt(r'D:\research\numerical optimization_imitation learning\nlp_nao_experiments\py_nlp_sdp_com.txt',outx, fmt="%3f")
    # np.savetxt(r'D:\research\numerical optimization_imitation learning\nlp_nao_experiments\py_kmp_foot_traj.txt',RLfoot_com_pos, fmt="%3f")


##    print "t:", time.clock()-start

if __name__ == "__main__":
    main()







