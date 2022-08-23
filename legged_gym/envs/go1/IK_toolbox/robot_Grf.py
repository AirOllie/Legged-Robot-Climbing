from __future__ import print_function

import zipapp

import pinocchio as pin
from pinocchio.explog import log
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import math
import copy
import sys
import mosek
import cvxpy as cp

#### Task-space force control:
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

class Force_dist:
    def __init__(self, urbodx):
        self.robot = urbodx
        self.g = 9.80
        self.mass= urbodx.getRobotMass()
        self.body_I = np.array([[0.0168352186,0.0004636141,0.0002367952],[0.0004636141,0.0656071082,3.6671e-05],[0.0002367952,3.6671e-05,0.0742720659]])

        #### leg force on each leg
        self.FR_force = np.zeros([3,1])
        self.FL_force = np.zeros([3,1])
        self.RR_force = np.zeros([3,1])
        self.RL_force = np.zeros([3,1])

        ### virtual leg pair force
        self.left_force = np.zeros([6,1])
        self.right_force = np.zeros([6, 1])

        self.F_Tor_tot = np.zeros([6, 1])
        self.co_Force = np.zeros([6, 6])


        ##### QP parameters
        self.qp_alpha = 10000
        self.qp_beta = 100
        self.qp_gama = 100
        self.fz_max = 100
        self.mu = 0.25
        self.x_offset = 0.000

        ##### QP formulation: unknown x
        ##### QP 1/2 X^𝑇 𝑄_𝑜 X + 𝑐^𝑇 X + 𝑐_𝑓
        ##### s.t. H X <= L
        self.var_num = 12
        self.con_num = 24
        self.qp_Q0 = np.zeros([self.var_num,self.var_num])
        self.qp_c = np.zeros([self.var_num,1])

        self.qp_H = np.zeros([self.con_num,self.var_num])
        self.qp_L = np.zeros([self.con_num,1])

        #### optimal leg force on each leg
        self.leg_force_opt = np.zeros([self.var_num, 1])
        self.leg_force_opt_ref = np.zeros([self.var_num, 1])
        self.leg_force_opt_old = np.zeros([self.var_num, 1])

    def Grf_ref_pre(self,gait_mode,leg_support,base_p,base_acc,base_r,base_r_acc,FR_p,FL_p,RR_p,RL_p,R_leg_p,L_leg_p,cop_xyz):
        self.FR_force = np.zeros([3,1])
        self.FL_force = np.zeros([3,1])
        self.RR_force = np.zeros([3,1])
        self.RL_force = np.zeros([3,1])

        base_acc[2,0] += self.g
        F_total = self.mass * base_acc

        Body_Rmatrix = self.RotMatrixfromEuler(base_r)
        Global_I = np.dot(np.dot(Body_Rmatrix, self.body_I), Body_Rmatrix.T)
        Torque_total = np.dot(Global_I,base_r_acc)

        self.F_Tor_tot[0:3,0] = F_total[0:3,0]
        self.F_Tor_tot[3:6,0] = Torque_total[0:3, 0]
        left_right_force = np.zeros([6,1])

        ###### hueristic guess: com distributor
        base_p[0,0] = copy.deepcopy(base_p[0,0])+self.x_offset
        base_p[2,0] = (R_leg_p[2,0] + L_leg_p[2,0])/2
        if(gait_mode==1): ### troting gait pair:FR-RL-left; pair:FL-RR-right
            if(leg_support==0): #### left support
                ##### first choice: com-->support feet
                vect_com_foot = (base_p - FR_p).T
                vec_foot = RL_p - FR_p

                doct_pro = vect_com_foot.dot(vec_foot)
                # dis_com_foot = np.linalg.norm(vect_com_foot)
                dis_foot = np.linalg.norm(vec_foot)

                force_alpha = doct_pro/((dis_foot)**2)
                lamda = self.clamp_function(force_alpha)

                self.FR_force[0:3,0] = (1 - lamda) * self.F_Tor_tot[0:3,0]
                self.RL_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]

            elif(leg_support==1): ### right support
                ##### first choice: com-->support feet
                vect_com_foot = (base_p - FL_p).T
                vec_foot = RR_p - FL_p

                doct_pro = vect_com_foot.dot(vec_foot)
                # dis_com_foot = np.linalg.norm(vect_com_foot)
                dis_foot = np.linalg.norm(vec_foot)

                force_alpha = doct_pro/((dis_foot)**2)
                lamda = self.clamp_function(force_alpha)

                self.FL_force[0:3, 0] = (1 - lamda) * self.F_Tor_tot[0:3,0]
                self.RR_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]

            else:

                vect_com_foot = (base_p - L_leg_p).T
                vec_foot = R_leg_p - L_leg_p

                doct_pro = vect_com_foot.dot(vec_foot)
                # dis_com_foot = np.linalg.norm(vect_com_foot)
                dis_foot = np.linalg.norm(vec_foot)

                force_alpha = doct_pro/((dis_foot)**2)
                lamda = self.clamp_function(force_alpha)

                # print(force_alpha)
                # print(vect_com_foot)
                # print(vec_foot)

                left_pair_force = (1 - lamda) * self.F_Tor_tot[0:3,0] ####left legs pair
                right_pair_force= lamda * self.F_Tor_tot[0:3, 0] ####right legs pair

                ##### Further force distrubition to left leg pair
                vect_com_foot = (L_leg_p - FR_p).T
                vec_foot = RL_p - FR_p

                doct_pro = vect_com_foot.dot(vec_foot)
                # dis_com_foot = np.linalg.norm(vect_com_foot)
                dis_foot = np.linalg.norm(vec_foot)

                force_alpha = doct_pro/((dis_foot)**2)
                lamda = self.clamp_function(force_alpha)
                # print(lamda)
                self.FR_force[0:3,0] = (1 - lamda) * left_pair_force
                self.RL_force[0:3, 0] = lamda * left_pair_force

                ##### Further force distrubition to right leg pair
                vect_com_foot = (R_leg_p - FL_p).T
                vec_foot = RR_p - FL_p

                doct_pro = vect_com_foot.dot(vec_foot)
                # dis_com_foot = np.linalg.norm(vect_com_foot)
                dis_foot = np.linalg.norm(vec_foot)

                force_alpha = doct_pro/((dis_foot)**2)
                lamda = self.clamp_function(force_alpha)
                # print(lamda)
                self.FL_force[0:3, 0] = (1 - lamda) * right_pair_force
                self.RR_force[0:3, 0] = lamda * right_pair_force
        else:
            if(gait_mode==0): ###pace gait pair:FL-RL-left; pair:FR-RR-right
                if(leg_support==0): #### left support
                    ##### first choice: com-->support feet
                    vect_com_foot = (base_p - FL_p).T
                    vec_foot = RL_p - FL_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro/((dis_foot)**2)
                    lamda = self.clamp_function(force_alpha)

                    self.FL_force[0:3,0] = (1 - lamda) * self.F_Tor_tot[0:3,0]
                    self.RL_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]

                elif(leg_support==1): ### right support
                    ##### first choice: com-->support feet
                    vect_com_foot = (base_p - FR_p).T
                    vec_foot = RR_p - FR_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro/((dis_foot)**2)
                    lamda = self.clamp_function(force_alpha)

                    self.FR_force[0:3, 0] = (1 - lamda) * self.F_Tor_tot[0:3,0]
                    self.RR_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]

                else:

                    vect_com_foot = (base_p - L_leg_p).T
                    vec_foot = R_leg_p - L_leg_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro/((dis_foot)**2)
                    lamda = self.clamp_function(force_alpha)

                    # print(force_alpha)
                    # print(vect_com_foot)
                    # print(vec_foot)

                    left_pair_force = (1 - lamda) * self.F_Tor_tot[0:3,0] ####left legs pair
                    right_pair_force= lamda * self.F_Tor_tot[0:3, 0] ####right legs pair

                    ##### Further force distrubition to left leg pair
                    vect_com_foot = (L_leg_p - FL_p).T
                    vec_foot = RL_p - FL_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro/((dis_foot)**2)
                    lamda = self.clamp_function(force_alpha)
                    # print(lamda)
                    self.FL_force[0:3,0] = (1 - lamda) * left_pair_force
                    self.RL_force[0:3, 0] = lamda * left_pair_force

                    ##### Further force distrubition to right leg pair
                    vect_com_foot = (R_leg_p - FR_p).T
                    vec_foot = RR_p - FR_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro/((dis_foot)**2)
                    lamda = self.clamp_function(force_alpha)
                    # print(lamda)
                    self.FR_force[0:3, 0] = (1 - lamda) * right_pair_force
                    self.RR_force[0:3, 0] = lamda * right_pair_force
            else:  #### bounding :FL--FR-left pair; RL--RR-right pair
                print("Grf initial distribution")
                if (leg_support == 0):  #### left support
                    ##### first choice: com-->support feet
                    vect_com_foot = (base_p - FL_p).T
                    vec_foot = FR_p - FL_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro / ((dis_foot) ** 2)
                    lamda = self.clamp_function(force_alpha)

                    self.FL_force[0:3, 0] = (1 - lamda) * self.F_Tor_tot[0:3, 0]
                    self.FR_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]
                elif (leg_support == 1):  ### right support
                    ##### first choice: com-->support feet
                    vect_com_foot = (base_p - RR_p).T
                    vec_foot = RL_p - RR_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro / ((dis_foot) ** 2)
                    lamda = self.clamp_function(force_alpha)

                    self.RR_force[0:3, 0] = (1 - lamda) * self.F_Tor_tot[0:3, 0]
                    self.RL_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]

                else:
                    vect_com_foot = (base_p - L_leg_p).T
                    vec_foot = R_leg_p - L_leg_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro / ((dis_foot) ** 2)
                    lamda = self.clamp_function(force_alpha)

                    left_pair_force = (1 - lamda) * self.F_Tor_tot[0:3, 0]  ####left legs pair
                    right_pair_force = lamda * self.F_Tor_tot[0:3, 0]  ####right legs pair

                    ##### Further force distrubition to left leg pair
                    vect_com_foot = (L_leg_p - FL_p).T
                    vec_foot = FR_p - FL_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro / ((dis_foot) ** 2)
                    lamda = self.clamp_function(force_alpha)
                    # print(lamda)
                    self.FL_force[0:3, 0] = (1 - lamda) * left_pair_force
                    self.FR_force[0:3, 0] = lamda * left_pair_force

                    ##### Further force distrubition to right leg pair
                    vect_com_foot = (R_leg_p - RR_p).T
                    vec_foot = RL_p - RR_p

                    doct_pro = vect_com_foot.dot(vec_foot)
                    # dis_com_foot = np.linalg.norm(vect_com_foot)
                    dis_foot = np.linalg.norm(vec_foot)

                    force_alpha = doct_pro / ((dis_foot) ** 2)
                    lamda = self.clamp_function(force_alpha)
                    # print(lamda)
                    self.RR_force[0:3, 0] = (1 - lamda) * right_pair_force
                    self.RL_force[0:3, 0] = lamda * right_pair_force

        ################# the following section result in infinite solutions##########################
        # if(gait_mode==1): ### troting gait pair:FRRL-left; pair:FLRR-right
        #     if(leg_support==0): #### left support
        #         co_R_1 = np.array([[1, 0, 0, 1, 0, 0]])
        #         co_R_2 = np.array([[0, 1, 0, 0, 1, 0]])
        #         co_R_3 = np.array([[0, 0, 1, 0, 0, 1]])
        #         vect_com_f1 = (base_p - FR_p).T
        #         vect_com_f2 = (base_p - RL_p).T
        #         # note a bug that they would be infinite solution for cross product,
        #         ######## here, we need a heuristic
        #         co_R_4 = np.array(
        #             [[0, -vect_com_f1[0, 2], vect_com_f1[0, 1], -0, -vect_com_f2[0, 2], vect_com_f2[0, 1]]])
        #         co_R_5 = np.array(
        #             [[vect_com_f1[0, 2], 0, -vect_com_f1[0, 0], vect_com_f2[0, 2], -0, -vect_com_f2[0, 0]]])
        #         co_R_6 = np.array(
        #             [[-vect_com_f1[0, 1], vect_com_f1[0, 0], 0, -vect_com_f2[0, 1], vect_com_f2[0, 0], -0]])
        #
        #
        #         self.co_Force[0, :] = co_R_1[0, :]
        #         self.co_Force[1, :] = co_R_2[0, :]
        #         self.co_Force[2, :] = co_R_3[0, :]
        #         self.co_Force[3, :] = co_R_4[0, :]
        #         self.co_Force[4, :] = co_R_5[0, :]
        #         self.co_Force[5, :] = co_R_6[0, :]
        #
        #         # print(self.co_Force)
        #         self.left_force = np.dot(np.linalg.inv(self.co_Force),self.F_Tor_tot)
        #
        #         self.FR_force[0:3,0] = self.left_force[0:3,0]
        #         self.RL_force[0:3, 0] = self.left_force[3:6, 0]
        #
        #     elif(leg_support==1): ### right support
        #         co_R_1 = np.array([[1, 0, 0, 1, 0, 0]])
        #         co_R_2 = np.array([[0, 1, 0, 0, 1, 0]])
        #         co_R_3 = np.array([[0, 0, 1, 0, 0, 1]])
        #         vect_com_f1 = (base_p - FL_p).T
        #         vect_com_f2 = (base_p - RR_p).T
        #         # vect_com_f1 = (base_p - FR_p).T
        #         # vect_com_f2 = (base_p - RR_p).T
        #         co_R_4 = np.array([[            0, -vect_com_f1[0,2],  vect_com_f1[0,1],            -0, -vect_com_f2[0,2],  vect_com_f2[0,1]]])
        #         co_R_5 = np.array([[ vect_com_f1[0,2],             0, -vect_com_f1[0,0], vect_com_f2[0,2],              -0, -vect_com_f2[0,0]]])
        #         co_R_6 = np.array([[-vect_com_f1[0,1],  vect_com_f1[0,0],              0, -vect_com_f2[0,1],  vect_com_f2[0,0],             -0]])
        #         self.co_Force[0, :] = co_R_1[0, :]
        #         self.co_Force[1, :] = co_R_2[0, :]
        #         self.co_Force[2, :] = co_R_3[0, :]
        #         self.co_Force[3, :] = co_R_4[0, :]
        #         self.co_Force[4, :] = co_R_5[0, :]
        #         self.co_Force[5, :] = co_R_6[0, :]
        #
        #         self.right_force = np.dot(np.linalg.inv(self.co_Force),self.F_Tor_tot)
        #         self.FL_force[0:3, 0] = self.right_force[0:3,  0]
        #         self.RR_force[0:3, 0] = self.right_force[3:6,  0]
        #         # self.FR_force[0:3, 0] = self.right_force[0:3,  0]
        #         # self.RR_force[0:3, 0] = self.right_force[3:6,  0]
        #     else:
        #         co_R_1 = np.array([[1, 0, 0, 1, 0, 0]])
        #         co_R_2 = np.array([[0, 1, 0, 0, 1, 0]])
        #         co_R_3 = np.array([[0, 0, 1, 0, 0, 1]])
        #         vect_com_f1 = (base_p - L_leg_p).T
        #         vect_com_f2 = (base_p - R_leg_p).T
        #         co_R_4 = np.array(
        #             [[0, -vect_com_f1[0, 2], vect_com_f1[0, 1], -0, -vect_com_f2[0, 2], vect_com_f2[0, 1]]])
        #         co_R_5 = np.array(
        #             [[vect_com_f1[0, 2], 0, -vect_com_f1[0, 0], vect_com_f2[0, 2], -0, -vect_com_f2[0, 0]]])
        #         co_R_6 = np.array(
        #             [[-vect_com_f1[0, 1], vect_com_f1[0, 0], 0, -vect_com_f2[0, 1], vect_com_f2[0, 0], -0]])
        #
        #
        #         self.co_Force[0, :] = co_R_1[0, :]
        #         self.co_Force[1, :] = co_R_2[0, :]
        #         self.co_Force[2, :] = co_R_3[0, :]
        #         self.co_Force[3, :] = co_R_4[0, :]
        #         self.co_Force[4, :] = co_R_5[0, :]
        #         self.co_Force[5, :] = co_R_6[0, :]
        #
        #         # print(np.linalg.det(self.co_Force))
        #         try:
        #             left_right_force = np.dot(np.linalg.inv(self.co_Force),self.F_Tor_tot)
        #         except:
        #             left_right_force[0:3, 0] = self.F_Tor_tot[0:3, 0]/2
        #             left_right_force[3:6, 0] = self.F_Tor_tot[0:3, 0] / 2
        #
        #
        #         left_pair_force = np.zeros([6, 1])
        #         right_pair_force = np.zeros([6, 1])
        #         left_pair_force[0:3, 0] = left_right_force[0:3,  0] ####left legs pair
        #         right_pair_force[0:3, 0] = left_right_force[3:6,  0]####right legs pair
        #
        #         ##### Further force distrubition to left leg pair
        #         co_R_1 = np.array([[1, 0, 0, 1, 0, 0]])
        #         co_R_2 = np.array([[0, 1, 0, 0, 1, 0]])
        #         co_R_3 = np.array([[0, 0, 1, 0, 0, 1]])
        #         vect_com_f1 = (L_leg_p - FR_p).T
        #         vect_com_f2 = (L_leg_p - RL_p).T
        #         # vect_com_f1 = (L_leg_p - FL_p).T
        #         # vect_com_f2 = (L_leg_p - RL_p).T
        #         co_R_4 = np.array(
        #             [[0, -vect_com_f1[0, 2], vect_com_f1[0, 1], -0, -vect_com_f2[0, 2], vect_com_f2[0, 1]]])
        #         co_R_5 = np.array(
        #             [[vect_com_f1[0, 2], 0, -vect_com_f1[0, 0], vect_com_f2[0, 2], -0, -vect_com_f2[0, 0]]])
        #         co_R_6 = np.array(
        #             [[-vect_com_f1[0, 1], vect_com_f1[0, 0], 0, -vect_com_f2[0, 1], vect_com_f2[0, 0], -0]])
        #
        #
        #         self.co_Force[0, :] = co_R_1[0, :]
        #         self.co_Force[1, :] = co_R_2[0, :]
        #         self.co_Force[2, :] = co_R_3[0, :]
        #         self.co_Force[3, :] = co_R_4[0, :]
        #         self.co_Force[4, :] = co_R_5[0, :]
        #         self.co_Force[5, :] = co_R_6[0, :]
        #
        #         self.left_force = np.dot(np.linalg.inv(self.co_Force),left_pair_force)
        #         self.FR_force[0:3,0] = self.left_force[0:3,0]
        #         self.RL_force[0:3, 0] = self.left_force[3:6, 0]
        #         # self.FL_force[0:3,0] = self.left_force[0:3,0]
        #         # self.RL_force[0:3, 0] = self.left_force[3:6, 0]
        #         ##### Further force distrubition to right leg pair
        #         co_R_1 = np.array([[1, 0, 0, 1, 0, 0]])
        #         co_R_2 = np.array([[0, 1, 0, 0, 1, 0]])
        #         co_R_3 = np.array([[0, 0, 1, 0, 0, 1]])
        #         vect_com_f1 = (R_leg_p - FL_p).T
        #         vect_com_f2 = (R_leg_p - RR_p).T
        #         # vect_com_f1 = (R_leg_p - FR_p).T
        #         # vect_com_f2 = (R_leg_p - RR_p).T
        #         co_R_4 = np.array(
        #             [[0, -vect_com_f1[0, 2], vect_com_f1[0, 1], -0, -vect_com_f2[0, 2], vect_com_f2[0, 1]]])
        #         co_R_5 = np.array(
        #             [[vect_com_f1[0, 2], 0, -vect_com_f1[0, 0], vect_com_f2[0, 2], -0, -vect_com_f2[0, 0]]])
        #         co_R_6 = np.array(
        #             [[-vect_com_f1[0, 1], vect_com_f1[0, 0], 0, -vect_com_f2[0, 1], vect_com_f2[0, 0], -0]])
        #
        #
        #         self.co_Force[0, :] = co_R_1[0, :]
        #         self.co_Force[1, :] = co_R_2[0, :]
        #         self.co_Force[2, :] = co_R_3[0, :]
        #         self.co_Force[3, :] = co_R_4[0, :]
        #         self.co_Force[4, :] = co_R_5[0, :]
        #         self.co_Force[5, :] = co_R_6[0, :]
        #         self.right_force = np.dot(np.linalg.inv(self.co_Force),right_pair_force)
        #         self.FR_force[0:3,0] = self.right_force[0:3,0]
        #         self.RL_force[0:3, 0] = self.right_force[3:6, 0]
        #         # self.FR_force[0:3,0] = self.right_force[0:3,0]
        #         # self.RR_force[0:3, 0] = self.right_force[3:6, 0]

        return self.F_Tor_tot, self.FR_force, self.FL_force, self.RR_force, self.RL_force

    ###### GRF optmization ##############################
    def Grf_ref_opt(self,gait_mode,leg_support,base_px,FR_p,FL_p,RR_p,RL_p):
        base_p = copy.deepcopy(base_px)
        base_p[0,0] = copy.deepcopy(base_px[0,0])+self.x_offset
        #### note: AF = d_total
        A = np.zeros([6,12])
        A[0:3, 0:3] = np.eye(3)
        A[0:3, 3:6] = np.eye(3)
        A[0:3, 6:9] = np.eye(3)
        A[0:3, 9:12] = np.eye(3)

        com_fr = base_p - FR_p
        w_hat = self.skew_hat(com_fr)
        A[3:6, 0:3] = w_hat

        com_fl = base_p - FL_p
        w_hat = self.skew_hat(com_fl)
        A[3:6, 3:6] = w_hat

        com_rr = base_p - RR_p
        w_hat = self.skew_hat(com_rr)
        A[3:6, 6:9] = w_hat

        com_rl = base_p - RL_p
        w_hat = self.skew_hat(com_rl)
        A[3:6, 9:12] = w_hat

        qp_Q0 = 2 * (self.qp_alpha * np.dot(A.T,A) + (self.qp_beta + self.qp_gama) * np.eye(self.var_num) )
        self.qp_Q0 = 0.5 * (qp_Q0.T + qp_Q0) ###


        #### ref
        self.leg_force_opt_ref[0:3, 0] = self.FR_force[0:3,0]
        self.leg_force_opt_ref[3:6, 0] = self.FL_force[0:3,0]
        self.leg_force_opt_ref[6:9, 0] = self.RR_force[0:3,0]
        self.leg_force_opt_ref[9:12, 0] = self.RL_force[0:3,0]

        self.qp_c = -2 * (self.qp_alpha * np.dot(A.T, self.F_Tor_tot) +
                          self.qp_beta * self.leg_force_opt_ref + self.qp_gama * self.leg_force_opt_old)

        ###### feasibility constraints##############
        self.qp_H = np.zeros([self.con_num,self.var_num])
        self.qp_L = np.zeros([self.con_num,1])
        ####  0<=fz<=fz_max
        for i in range(0,4):
            self.qp_H[2*i,3*i+2] = -1
            self.qp_H[2*i+1,3*i+2] = 1
            self.qp_L[2*i+1,0] = self.fz_max
            # print(self.qp_L)
        ####  -u f_z =< f_x <= u f_z
        for i in range(0,4):
            self.qp_H[8+2*i,3*i] = -1
            self.qp_H[8+2*i,3*i+2] = -self.mu
            self.qp_H[8+2*i+1,3*i] = 1
            self.qp_H[8+2*i+1,3*i+2] = -self.mu

        for i in range(0,4):
            self.qp_H[16+2*i,3*i+1] = -1
            self.qp_H[16+2*i,3*i+2] = -self.mu
            self.qp_H[16+2*i+1,3*i+1] = 1
            self.qp_H[16+2*i+1,3*i+2] = -self.mu

        ##### equality constraints:
        AA = np.zeros([self.var_num,self.var_num])

        if(gait_mode==1): ### troting gait pair:FRRL-left; pair:FLRR-right
            if(leg_support==0): #### left support, right two legs zeros force
                AA[3:6,3:6] = np.eye(3)
                AA[6:9,6:9] = np.eye(3)
                # self.FR_force[0:3,0] = (1 - lamda) * self.F_Tor_tot[0:3,0]
                # self.RL_force[0:3, 0] = lamda * self.F_Tor_tot[0:3, 0]
            elif(leg_support==1): ### right support
                AA[0:3,0:3] = np.eye(3)
                AA[9:12,9:12] = np.eye(3)
        else:
            if(gait_mode==0): ###pace gait pair:FL-RL-left; pair:FR-RR-right
                if (leg_support == 0):  #### left support, right two legs zeros force
                    AA[0:3, 0:3] = np.eye(3)
                    AA[6:9, 6:9] = np.eye(3)
                elif (leg_support == 1):  ### right support
                    AA[3:6, 3:6] = np.eye(3)
                    AA[9:12, 9:12] = np.eye(3)
            else:  #### bounding :FL--FR-left pair; RL--RR-right pair
                print("bounding gait grf optimization")
                if (leg_support == 0):  #### left support, right two legs zeros force
                    AA[6:9, 6:9] = np.eye(3)
                    AA[9:12, 9:12] = np.eye(3)
                elif (leg_support == 1):  ### right support
                    AA[0:3, 0:3] = np.eye(3)
                    AA[3:6, 3:6] = np.eye(3)

        bb = np.zeros([self.var_num, 1])


        if(np.linalg.norm(self.leg_force_opt_old)==0): ###first time: only
            self.leg_force_opt = self.leg_force_opt_ref
        else:
            xx = cp.Variable([self.var_num,1])
            prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(xx, self.qp_Q0) + (self.qp_c).T @ xx),
                              [self.qp_H @ xx <= self.qp_L,
                               AA @ xx == bb])
            prob.solve()
            self.leg_force_opt = xx.value
            # print(A)



            # ### mosek
            # # Make mosek environment
            # with mosek.Env() as env:
            #     env.set_Stream(mosek.streamtype.log, streamprinter)
            #     # Create a task object and attach log stream printer
            #     with env.Task() as task:
            #         task.set_Stream(mosek.streamtype.log, streamprinter)
            #         numcon = self.con_num
            #         numvar = self.var_num
            #
            #         # Append 'numcon' empty constraints.
            #         task.appendcons(numcon)
            #         # Append matrix variables of sizes in 'numvar'.
            #         task.appendvars(numvar)
            #
            #
            #
            #         for i in range(numvar):
            #             # Set the linear term C in the objective.
            #             print(self.qp_c[i,0])
            #             task.putcj(i, self.qp_c[i,0])
            #
            #         inf = 10000000.0
            #         for i in range(numcon):
            #             bkc = mosek.boundkey.up
            #             task.putconbound(i, bkc, -inf, self.qp_L[i,0])
            #
            #             asub = []
            #             aval = []
            #             for j in range(0, numvar):
            #                 if(abs(self.qp_H[i,j])>0.0000000001):
            #                     asub.append(j)
            #                     aval.append(self.qp_H[i,j])
            #
            #             task.putarow(i,asub,aval)
            #
            #         qsubi, qsubj, qval = self.sparse_matrix_express(self.qp_Q0)
            #         task.putqobj(qsubi, qsubj, qval)
            #
            #         # Input the objective sense (minimize/maximize)
            #         task.putobjsense(mosek.objsense.minimize)
            #
            #         # Solve the problem and print summary
            #         task.optimize()
            #
            #         task.solutionsummary(mosek.streamtype.msg)
            #         # Get status information about the solution
            #         prosta = task.getprosta(mosek.soltype.itr)
            #         solsta = task.getsolsta(mosek.soltype.itr)
            #
            #         xx = [0.] * numvar
            #         task.getxx(mosek.soltype.itr,
            #                    xx)
            #         if (solsta == mosek.solsta.optimal):
            #             print("Optimal solution: %s" % xx)
            #             # print("Total force: %s" % self.F_Tor_tot)
            #             # print("leg_force_opt_ref: %s" % self.leg_force_opt_ref)
            #             # print("leg_force_opt_old: %s" % self.leg_force_opt_old)
            #             # print(self.qp_c)
            #         else:
            #             print('No optimal solutin using mosek!!!\n')
            #             self.leg_force_opt = self.leg_force_opt_ref

        self.leg_force_opt_old = self.leg_force_opt

        return self.leg_force_opt

    #### sparse_matrix_express
    def sparse_matrix_express(self, W):
        a = W.shape[0]
        b = W.shape[1]
        barci = []
        barcj = []
        barcval = []
        for i in range(0, a):
            for j in range(0, i + 1):
                if (np.abs(W[i, j]) > 0.000000000001):
                    barci.append(i)
                    barcj.append(j)
                    barcval.append(W[i, j])
        return barci, barcj, barcval



    ####### Rotation matrix generated by the  rpy angle
    def RotMatrixfromEuler(self, xyz):
        x_angle = xyz[0,0]
        y_angle = xyz[1,0]
        z_angle = xyz[2,0]
        Rrpy = np.array([[math.cos(y_angle) * math.cos(z_angle), math.cos(z_angle) * math.sin(x_angle) * \
                          math.sin(y_angle) - math.cos(x_angle) * math.sin(z_angle),
                          math.sin(x_angle) * math.sin(z_angle) + \
                          math.cos(x_angle) * math.cos(z_angle) * math.sin(y_angle)],
                         [math.cos(y_angle) * math.sin(z_angle), math.cos(x_angle) * math.cos(z_angle) + \
                          math.sin(x_angle) * math.sin(y_angle) * math.sin(z_angle),
                          math.cos(x_angle) * math.sin(y_angle) * math.sin(z_angle) \
                          - math.cos(z_angle) * math.sin(x_angle)],
                         [-math.sin(y_angle), math.cos(y_angle) * math.sin(x_angle),
                          math.cos(x_angle) * math.cos(y_angle)]])
        # qua_base = pybullet.getQuaternionFromEuler(xyz)
        # Rrpy = pybullet.getMatrixFromQuaternion(qua_base)

        return Rrpy

    def clamp_function(self,lamda):
        if(lamda<=0.2):
            lamda = 0.2
        elif(lamda>=0.8):
            lamda=0.8

        return lamda

    def skew_hat(self,vec_w):
        w_hat = np.array([[0,           -vec_w[2,0],  vec_w[1,0]],
                          [vec_w[2,0],            0, -vec_w[0,0]],
                          [-vec_w[1,0],  vec_w[0,0],           0]])
        return w_hat

