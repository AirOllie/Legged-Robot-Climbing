from __future__ import print_function

import pinocchio as pin
from pinocchio.explog import log
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import math
import copy

#### Task-space force control:

class ForceCtrl:
    def __init__(self, urbodx):
        self.robot = urbodx

        ### 250Hz
        ###### support leg
        self.kp = np.array([250, 350, 350])
        self.kd = np.array([0.5, 2, 2])
        ###### swing leg
        self.kp_swing = np.array([250, 350, 350])
        self.kd_swing = np.array([0.5, 2, 2])

        ##### swing leg impedance control
        self.kpCartesian = np.array([250, 250, 350])
        self.kdCartesian = np.array([0.5, 0.5, 2])



        # ### 200hz
        # ###### support leg
        # self.kp = np.array([100.0, 150.0, 150.0])
        # self.kd = np.array([0.05, 0.05, 0.05])
        # ###### swing leg
        # self.kp_swing = np.array([50.0, 150.0, 150.0])
        # self.kd_swing = np.array([0.5, 0.5, 0.5])
        # ##### swing leg impedance control
        # self.kpCartesian = np.array([10, 10, 10])
        # self.kdCartesian = np.array([0.1, 0.1, 0.5])
    def torque_cmd(self,leg_support,J_leg, F_leg,q_ref, q_mea, dq_ref, dq_mea,p_ref, p_mea, dp_ref, dp_mea,Gravity_comp):
        torque_cmd = np.array([0,0,0])
        ###### Noting: it should be noted that we can still take the stance leg as swing leg through considering the relative motion to CoM
        if leg_support: ###stance leg
            torque_ff = self.stance_leg_forward(J_leg, F_leg)
        else: ###swing leg
            torque_ff = self.swing_leg_forward(J_leg, p_ref, p_mea, dp_ref, dp_mea)

        joint_track_torque = self.Joint_track_fb(q_ref, q_mea, dq_ref, dq_mea,leg_support)
        Gra_comp = self.Gravity_comp(Gravity_comp)

        for i in range(0,3):
            #### feedforward plus feedback
            if leg_support: ####stance phase
                torque_cmd[i] = joint_track_torque[i] + torque_ff[i, 0]
                # torque_cmd[i] = joint_track_torque[i]  #### only feedback
            else:
                torque_cmd[i] = joint_track_torque[i] + torque_ff[i, 0] + Gra_comp[i]
                # torque_cmd[i] = joint_track_torque[i] + Gra_comp[i]

            # torque_cmd[i] = joint_track_torque[i] #### only feedback
            #torque_cmd[i] = torque_ff[i, 0]      #### only feedforward (using the real angle to compute Jacobian or not) not working, it turns out that the joint angle Feedback is indispensible
        if(torque_cmd[0]>10):
            torque_cmd[0] = 10
        elif(torque_cmd[0]<-10):
            torque_cmd[0] = -10

        if(torque_cmd[1]>10):
            torque_cmd[1] = 10
        elif(torque_cmd[1]<-10):
            torque_cmd[1] = -10

        if(torque_cmd[2]>10):
            torque_cmd[2] = 10
        elif(torque_cmd[2]<-10):
            torque_cmd[2] = -10

        return torque_cmd,joint_track_torque



    # FR, FL, RR, RL
    def stance_leg_forward(self, J_leg, F_leg):
        torque_ff = -np.dot(J_leg.T, F_leg)
        # print(torque_ff)
        return torque_ff

    def swing_leg_forward(self, J_leg, p_ref, p_mea, dp_ref, dp_mea):
        for i in range(0,3):
            dp_ref[i] = 0
        F_statence = self.kpCartesian * (p_ref - p_mea) + self.kdCartesian * (dp_ref - dp_mea)
        F_det = np.zeros([3,1])
        F_det[0, 0] = F_statence[0]
        F_det[1, 0] = F_statence[1]
        F_det[2, 0] = F_statence[2]
        #print("F_det:",F_det.T)
        torque_ff = np.dot(J_leg.T, F_det)
        #print("torque_ff",torque_ff.T)
        return torque_ff

    def Joint_track_fb(self,q_ref, q_mea, dq_ref, dq_mea,leg_support):
        for i in range(0,3):
            dq_ref[i] = 0
        if leg_support:
            torque_fb = self.kp * (q_ref - q_mea) + self.kd * (dq_ref - dq_mea)
        else:
            torque_fb = self.kp_swing * (q_ref - q_mea) + self.kd_swing * (dq_ref - dq_mea)
        #print("torque_fb:",torque_fb.T)
        return torque_fb

    def Gravity_comp(self, compensation):
        torque_gf = compensation
        #print("torque_gf:",torque_gf)
        return torque_gf
