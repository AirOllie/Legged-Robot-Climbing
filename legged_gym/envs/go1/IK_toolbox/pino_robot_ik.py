from __future__ import print_function

import pinocchio as pin
from pinocchio.explog import log
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import math
import copy

#### Jacobian based Ik
#### test
class CLIK:
    def __init__(self, robot, oMdes, index,Freebase):
        self.oMdes = oMdes
        self.index = index
        self.Freebase = Freebase
        self.robot = robot

        ### go1_parameters
        self.FR_leg_offset_x = 0.1881
        self.FL_leg_offset_x = 0.1881
        self.RR_leg_offset_x = -0.1881
        self.RL_leg_offset_x = -0.1881

        self.FR_leg_offset_y = self.RR_leg_offset_y = -0.04675
        self.FL_leg_offset_y = self.RL_leg_offset_y = 0.04675
        self.FR_thigh_y = self.RR_thigh_y = -0.08
        self.FL_thigh_y = self.RL_thigh_y = 0.08
        self.FR_thigh_length = self.FL_thigh_length = self.RR_thigh_length = self.RL_thigh_length =  -0.213
        self.FR_calf_length = self.FL_calf_length = self.RR_calf_length = self.RL_calf_length = -0.213

    # FR, FL, RR, RL
    def fk_close_form(self, body_P, body_R, q_joint, feet_flag):
        if(feet_flag == 0):
            leg_offset_x = self.FR_leg_offset_x
            leg_offset_y = self.FR_leg_offset_y
            thigh_y = self.FR_thigh_y
            thigh_length = self.FR_thigh_length
            calf_length = self.FR_calf_length
        else:
            if(feet_flag == 1):
                leg_offset_x = self.FL_leg_offset_x
                leg_offset_y = self.FL_leg_offset_y
                thigh_y = self.FL_thigh_y
                thigh_length = self.FL_thigh_length
                calf_length = self.FL_calf_length
            else:
                if (feet_flag == 2):
                    leg_offset_x = self.RR_leg_offset_x
                    leg_offset_y = self.RR_leg_offset_y
                    thigh_y = self.RR_thigh_y
                    thigh_length = self.RR_thigh_length
                    calf_length = self.RR_calf_length
                else:
                    leg_offset_x = self.RL_leg_offset_x
                    leg_offset_y = self.RL_leg_offset_y
                    thigh_y = self.RL_thigh_y
                    thigh_length = self.RL_thigh_length
                    calf_length = self.RL_calf_length

        q_hip = q_joint[0]
        q_thigh = q_joint[1]
        q_calf = q_joint[2]

        body_px = body_P[0]
        body_py = body_P[1]
        body_pz = body_P[2]

        body_r = body_R[0,0]
        body_p = body_R[1,0]
        body_y = body_R[2,0]

        pos_hip = np.zeros([3,1])
        pos_thigh = np.zeros([3,1])
        pos_calf = np.zeros([3, 1])
        pos_feet = np.zeros([3, 1])
        Jacobian_kin = np.zeros([3, 3])
        pos_hip[0, 0] = body_px - leg_offset_y * (math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p)
                                                 * math.sin(body_r)) + leg_offset_x * math.cos(body_p) * math.cos(body_y)

        pos_hip[1, 0] = body_py + leg_offset_y * (math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r)
                                                 * math.sin(body_y)) + leg_offset_x * math.cos(body_p) * math.sin(body_y)

        pos_hip[1, 0] = body_pz - leg_offset_x * math.sin(body_p) + leg_offset_y * math.cos(body_p) * math.sin(body_r)

        pos_thigh[0, 0] = body_px - thigh_y * (math.cos(q_hip) * (math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))
                                              - math.sin(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p))) \
                         - leg_offset_y * (math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r)) + leg_offset_x * math.cos(body_p) * math.cos(body_y)

        pos_thigh[1, 0] = body_py + thigh_y * (math.cos(q_hip) * (math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y)) - math.sin(q_hip)
                                              * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y))) \
                         + leg_offset_y * (math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y)) + leg_offset_x * math.cos(body_p) * math.sin(body_y)

        pos_thigh[1, 0] = body_pz - leg_offset_x * math.sin(body_p) + thigh_y * (math.cos(body_p) * math.cos(body_r) * math.sin(q_hip) + math.cos(body_p)
                                                                                * math.cos(q_hip) * math.sin(body_r)) + leg_offset_y * math.cos(body_p) * math.sin(body_r)

        pos_calf[0, 0] = body_px - thigh_y*(math.cos(q_hip)*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r)) - math.sin(q_hip)*(math.sin(body_r)*math.sin(body_y) + math.cos(body_r)*math.cos(body_y)*math.sin(body_p))) + thigh_length*(math.cos(q_thigh)*(math.cos(q_hip)*(math.sin(body_r)*math.sin(body_y) + math.cos(body_r)*math.cos(body_y)*math.sin(body_p)) + math.sin(q_hip)*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r))) + math.cos(body_p)*math.cos(body_y)*math.sin(q_thigh)) - leg_offset_y*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r)) + leg_offset_x*math.cos(body_p)*math.cos(body_y)

        pos_calf[1, 0] = body_py + thigh_y*(math.cos(q_hip)*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y)) - math.sin(q_hip)*(math.cos(body_y)*math.sin(body_r) - math.cos(body_r)*math.sin(body_p)*math.sin(body_y))) - thigh_length*(math.cos(q_thigh)*(math.cos(q_hip)*(math.cos(body_y)*math.sin(body_r) - math.cos(body_r)*math.sin(body_p)*math.sin(body_y)) + math.sin(q_hip)*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y))) - math.cos(body_p)*math.sin(body_y)*math.sin(q_thigh)) + leg_offset_y*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y)) + leg_offset_x*math.cos(body_p)*math.sin(body_y)

        pos_calf[2, 0] = body_pz - thigh_length*(math.sin(body_p)*math.sin(q_thigh) - math.cos(q_thigh)*(math.cos(body_p)*math.cos(body_r)*math.cos(q_hip) - math.cos(body_p)*math.sin(body_r)*math.sin(q_hip))) - leg_offset_x*math.sin(body_p) + thigh_y*(math.cos(body_p)*math.cos(body_r)*math.sin(q_hip) + math.cos(body_p)*math.cos(q_hip)*math.sin(body_r)) + leg_offset_y*math.cos(body_p)*math.sin(body_r)

        pos_feet[0, 0] = body_px - thigh_y*(math.cos(q_hip)*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r)) - math.sin(q_hip)*(math.sin(body_r)*math.sin(body_y) + math.cos(body_r)*math.cos(body_y)*math.sin(body_p))) + thigh_length*(math.cos(q_thigh)*(math.cos(q_hip)*(math.sin(body_r)*math.sin(body_y) + math.cos(body_r)*math.cos(body_y)*math.sin(body_p)) + math.sin(q_hip)*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r))) + math.cos(body_p)*math.cos(body_y)*math.sin(q_thigh)) + calf_length*(math.cos(q_calf)*(math.cos(q_thigh)*(math.cos(q_hip)*(math.sin(body_r)*math.sin(body_y) + math.cos(body_r)*math.cos(body_y)*math.sin(body_p)) + math.sin(q_hip)*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r))) + math.cos(body_p)*math.cos(body_y)*math.sin(q_thigh)) - math.sin(q_calf)*(math.sin(q_thigh)*(math.cos(q_hip)*(math.sin(body_r)*math.sin(body_y) + math.cos(body_r)*math.cos(body_y)*math.sin(body_p)) + math.sin(q_hip)*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r))) - math.cos(body_p)*math.cos(body_y)*math.cos(q_thigh))) - leg_offset_y*(math.cos(body_r)*math.sin(body_y) - math.cos(body_y)*math.sin(body_p)*math.sin(body_r)) + leg_offset_x*math.cos(body_p)*math.cos(body_y)
        pos_feet[1, 0] = body_py + thigh_y*(math.cos(q_hip)*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y)) - math.sin(q_hip)*(math.cos(body_y)*math.sin(body_r) - math.cos(body_r)*math.sin(body_p)*math.sin(body_y))) - thigh_length*(math.cos(q_thigh)*(math.cos(q_hip)*(math.cos(body_y)*math.sin(body_r) - math.cos(body_r)*math.sin(body_p)*math.sin(body_y)) + math.sin(q_hip)*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y))) - math.cos(body_p)*math.sin(body_y)*math.sin(q_thigh)) - calf_length*(math.cos(q_calf)*(math.cos(q_thigh)*(math.cos(q_hip)*(math.cos(body_y)*math.sin(body_r) - math.cos(body_r)*math.sin(body_p)*math.sin(body_y)) + math.sin(q_hip)*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y))) - math.cos(body_p)*math.sin(body_y)*math.sin(q_thigh)) - math.sin(q_calf)*(math.sin(q_thigh)*(math.cos(q_hip)*(math.cos(body_y)*math.sin(body_r) - math.cos(body_r)*math.sin(body_p)*math.sin(body_y)) + math.sin(q_hip)*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y))) + math.cos(body_p)*math.cos(q_thigh)*math.sin(body_y))) + leg_offset_y*(math.cos(body_r)*math.cos(body_y) + math.sin(body_p)*math.sin(body_r)*math.sin(body_y)) + leg_offset_x*math.cos(body_p)*math.sin(body_y)
        pos_feet[2, 0] = body_pz - thigh_length*(math.sin(body_p)*math.sin(q_thigh) - math.cos(q_thigh)*(math.cos(body_p)*math.cos(body_r)*math.cos(q_hip) - math.cos(body_p)*math.sin(body_r)*math.sin(q_hip))) - leg_offset_x*math.sin(body_p) + thigh_y*(math.cos(body_p)*math.cos(body_r)*math.sin(q_hip) + math.cos(body_p)*math.cos(q_hip)*math.sin(body_r)) - calf_length*(math.cos(q_calf)*(math.sin(body_p)*math.sin(q_thigh) - math.cos(q_thigh)*(math.cos(body_p)*math.cos(body_r)*math.cos(q_hip) - math.cos(body_p)*math.sin(body_r)*math.sin(q_hip))) + math.sin(q_calf)*(math.cos(q_thigh)*math.sin(body_p) + math.sin(q_thigh)*(math.cos(body_p)*math.cos(body_r)*math.cos(q_hip) - math.cos(body_p)*math.sin(body_r)*math.sin(q_hip)))) + leg_offset_y*math.cos(body_p)*math.sin(body_r)

        Jacobian_kin[0,0] = thigh_y * (
                    math.cos(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))) + calf_length * (
                                         math.cos(q_calf) * math.cos(q_thigh) * (math.cos(q_hip) * (
                                             math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r)) - math.sin(
                                     q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(
                                     body_p))) - math.sin(q_calf) * math.sin(q_thigh) * (math.cos(q_hip) * (
                                             math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r)) - math.sin(
                                     q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(
                                     body_p)))) + thigh_length * math.cos(q_thigh) * (math.cos(q_hip) * (
                    math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r)) - math.sin(q_hip) * (
                                                                                             math.sin(body_r) * math.sin(
                                                                                         body_y) + math.cos(body_r) * math.cos(
                                                                                         body_y) * math.sin(body_p)))
        Jacobian_kin[0,1] = - thigh_length * (math.sin(q_thigh) * (
                    math.cos(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))) - math.cos(body_p) * math.cos(
            body_y) * math.cos(q_thigh)) - calf_length * (math.cos(q_calf) * (math.sin(q_thigh) * (
                    math.cos(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))) - math.cos(body_p) * math.cos(
            body_y) * math.cos(q_thigh)) + math.sin(q_calf) * (math.cos(q_thigh) * (
                    math.cos(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))) + math.cos(body_p) * math.cos(
            body_y) * math.sin(q_thigh)))
        Jacobian_kin[0,2] = -calf_length * (math.cos(q_calf) * (math.sin(q_thigh) * (
                    math.cos(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))) - math.cos(body_p) * math.cos(
            body_y) * math.cos(q_thigh)) + math.sin(q_calf) * (math.cos(q_thigh) * (
                    math.cos(q_hip) * (math.sin(body_r) * math.sin(body_y) + math.cos(body_r) * math.cos(body_y) * math.sin(body_p)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.sin(body_y) - math.cos(body_y) * math.sin(body_p) * math.sin(body_r))) + math.cos(body_p) * math.cos(
            body_y) * math.sin(q_thigh)))
        Jacobian_kin[1,0] = - thigh_y * (
                    math.cos(q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y))) - calf_length * (
                                         math.cos(q_calf) * math.cos(q_thigh) * (math.cos(q_hip) * (
                                             math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y)) - math.sin(
                                     q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(
                                     body_y))) - math.sin(q_calf) * math.sin(q_thigh) * (math.cos(q_hip) * (
                                             math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y)) - math.sin(
                                     q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(
                                     body_y)))) - thigh_length * math.cos(q_thigh) * (math.cos(q_hip) * (
                    math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y)) - math.sin(q_hip) * (
                                                                                             math.cos(body_y) * math.sin(
                                                                                         body_r) - math.cos(body_r) * math.sin(
                                                                                         body_p) * math.sin(body_y)))
        Jacobian_kin[1,1] = thigh_length * (math.sin(q_thigh) * (
                    math.cos(q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y))) + math.cos(body_p) * math.cos(
            q_thigh) * math.sin(body_y)) + calf_length * (math.cos(q_calf) * (math.sin(q_thigh) * (
                    math.cos(q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y))) + math.cos(body_p) * math.cos(
            q_thigh) * math.sin(body_y)) + math.sin(q_calf) * (math.cos(q_thigh) * (
                    math.cos(q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y))) - math.cos(body_p) * math.sin(
            body_y) * math.sin(q_thigh)))
        Jacobian_kin[1,2] = calf_length * (math.cos(q_calf) * (math.sin(q_thigh) * (
                    math.cos(q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y))) + math.cos(body_p) * math.cos(
            q_thigh) * math.sin(body_y)) + math.sin(q_calf) * (math.cos(q_thigh) * (
                    math.cos(q_hip) * (math.cos(body_y) * math.sin(body_r) - math.cos(body_r) * math.sin(body_p) * math.sin(body_y)) + math.sin(q_hip) * (
                        math.cos(body_r) * math.cos(body_y) + math.sin(body_p) * math.sin(body_r) * math.sin(body_y))) - math.cos(body_p) * math.sin(
            body_y) * math.sin(q_thigh)))
        Jacobian_kin[2,0] = thigh_y * (
                    math.cos(body_p) * math.cos(body_r) * math.cos(q_hip) - math.cos(body_p) * math.sin(body_r) * math.sin(q_hip)) - calf_length * (
                                         math.cos(q_calf) * math.cos(q_thigh) * (
                                             math.cos(body_p) * math.cos(body_r) * math.sin(q_hip) + math.cos(body_p) * math.cos(q_hip) * math.sin(
                                         body_r)) - math.sin(q_calf) * math.sin(q_thigh) * (
                                                     math.cos(body_p) * math.cos(body_r) * math.sin(q_hip) + math.cos(body_p) * math.cos(
                                                 q_hip) * math.sin(body_r))) - thigh_length * math.cos(q_thigh) * (
                                         math.cos(body_p) * math.cos(body_r) * math.sin(q_hip) + math.cos(body_p) * math.cos(q_hip) * math.sin(
                                     body_r))
        Jacobian_kin[2, 1] = - thigh_length * (math.cos(q_thigh) * math.sin(body_p) + math.sin(q_thigh) * (
                    math.cos(body_p) * math.cos(body_r) * math.cos(q_hip) - math.cos(body_p) * math.sin(body_r) * math.sin(q_hip))) - calf_length * (
                                         math.cos(q_calf) * (math.cos(q_thigh) * math.sin(body_p) + math.sin(q_thigh) * (
                                             math.cos(body_p) * math.cos(body_r) * math.cos(q_hip) - math.cos(body_p) * math.sin(body_r) * math.sin(
                                         q_hip))) - math.sin(q_calf) * (math.sin(body_p) * math.sin(q_thigh) - math.cos(q_thigh) * (
                                             math.cos(body_p) * math.cos(body_r) * math.cos(q_hip) - math.cos(body_p) * math.sin(body_r) * math.sin(
                                         q_hip))))
        Jacobian_kin[2, 2] = -calf_length * (math.cos(q_calf) * (math.cos(q_thigh) * math.sin(body_p) + math.sin(q_thigh) * (
                    math.cos(body_p) * math.cos(body_r) * math.cos(q_hip) - math.cos(body_p) * math.sin(body_r) * math.sin(q_hip))) - math.sin(q_calf) * (
                                                         math.sin(body_p) * math.sin(q_thigh) - math.cos(q_thigh) * (
                                                             math.cos(body_p) * math.cos(body_r) * math.cos(q_hip) - math.cos(body_p) * math.sin(
                                                         body_r) * math.sin(q_hip))))



        return pos_feet, Jacobian_kin

    def ik_close_form(self, body_P, body_R, p_des, q_ini,feet_flag, It_max =15, lamda=0.5):
        i=0
        pos_cal, Joc = self.fk_close_form(body_P, body_R, q_ini, feet_flag)
        q_des = copy.deepcopy(q_ini)

        err = np.zeros([3,1])

        while (i<It_max):
            err = p_des - pos_cal
            Joc_inv = np.linalg.inv(Joc)
            det_angle = lamda * np.dot(Joc_inv,err)

            q_des[0] += det_angle[0, 0]
            q_des[1] += det_angle[1, 0]
            q_des[2] += det_angle[2, 0]

            if((err[0,0]**2 + err[1,0]**2 + err[2,0]**2 ) < 1e-6):
                break

            pos_cal, Joc = self.fk_close_form(body_P, body_R, q_des, feet_flag)

            i += 1

        # print("*************************************************************************")
        return q_des, Joc


    #pinocchio-based Jacobian IK
    def ik_Jacobian(self, q, Freebase, eps= 1e-4,IT_MAX = 1000,DT = 1e-1,damp = 1e-6):
        i=0
        while True:
            pin.forwardKinematics(self.robot.model,self.robot.data,q)
            dMi = self.oMdes.actInv(self.robot.data.oMi[self.index])
            err = pin.log(dMi).vector
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            J = pin.computeJointJacobian(self.robot.model,self.robot.data,q,self.index)
            if Freebase:
                J[0:6,0:7] = np.zeros((6,7))
            v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            if Freebase:
                v[0:7] = np.zeros([7])
            q = pin.integrate(self.robot.model,q,v*DT)
            if Freebase:
                q[0:7] = np.zeros([7])

            # if not i % 10:
            #     print('%d: error = %s' % (i, err.T))
            i += 1
        J = pin.computeJointJacobian(self.robot.model, self.robot.data, q, self.index)
        # print("*************************************************************************")
        return q, J



    #pinocchio-based optimization IK
    def position_error(self, q):
        # ### considering the orn and pos: expected status
        no = log(self.oMdes)
        log_oMdes_vec = no.vector

        #### calculating the robot status:
        pin.forwardKinematics(self.robot.model, self.robot.data, q)

        p = self.robot.data.oMi[self.index]
        nv = log(p)
        log_po_vec = nv.vector

        err1 = np.sqrt((log_oMdes_vec[0] - log_po_vec[0]) ** 2 + (log_oMdes_vec[1] - log_po_vec[1]) ** 2 + (
                log_oMdes_vec[2] - log_po_vec[2]) ** 2)

        # err2 = np.sqrt((q[0]) ** 2 + (q[1]) ** 2 + (q[2]) ** 2 + (q[3]) ** 2 + 1000*(q[4]) ** 2)  ####base joint
        err = 100 * err1
        return err

    def fbgs_opt(self,q):
        # eerrr = self.position_error(q)
        xopt_bfgs = fmin_bfgs(self.position_error, q)
        #print('*** Xopt in BFGS =', xopt_bfgs[7:13])
        return xopt_bfgs

