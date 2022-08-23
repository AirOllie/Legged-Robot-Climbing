from __future__ import print_function

import pinocchio as pin
from pinocchio.explog import log
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import math
import copy

#### Motion planner
class Gait:
    def __init__(self, T = 0.69999, Sx = 0.08, Sy = 0, Sz = 0, lift = 0.03, T_num = 15, Dsp_ratio=0.1, dt = 0.005, Qudrupedal=True):
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

        # Sy = (self.FL_leg_offset_y + self.FL_thigh_y) *2
        ### gait_parameter
        self.T = T
        self.dt = dt
        self._footstepsnumber = T_num
        self.Dsp_ratio = Dsp_ratio
        self._steplength = np.ones([T_num,1])
        self._stepwidth = np.ones([T_num, 1])
        self._stepheight = np.ones([T_num, 1])
        self._lift_height_ref = np.ones([T_num, 1])

        self._steplength *= Sx
        self._steplength[0,0] = 0
        self._steplength[1,0] = 0
        self._steplength[2, 0] = Sx / 2
        self._steplength[4, 0] = 0
        self._steplength[5:9, :] = -Sx*np.ones([4,1])
        self._steplength[10, :] = 0
        self._steplength[11, :] = Sx / 2
        self._steplength[-1, :] = 0


        self._steplength[self._footstepsnumber - 1,0] = 0
        self._steplength[self._footstepsnumber - 2,0] = 0


        self._stepwidth *= ((self.FL_leg_offset_y + self.FL_thigh_y) *2)
        self._stepwidth[0,0] *= 0.5

        self._stepheight *= Sz

        self._lift_height_ref *= lift
        self._lift_height_ref[self._footstepsnumber-1,0] = 0
        self._lift_height_ref[self._footstepsnumber-2,0] = 0
        self._lift_height_ref[self._footstepsnumber-3,0] *= 0.5

        self._footx_ref = np.zeros([T_num,1])
        self._footy_ref = np.zeros([T_num,1])
        self._footz_ref = np.zeros([T_num,1])

        self.footx_real = np.zeros([T_num,1])
        self.footy_real = np.zeros([T_num,1])
        self.footz_real = np.zeros([T_num, 1])

        ### reference walking period

        self.Ts = T * np.ones([T_num, 1])
        self.Tx = np.zeros([T_num, 1])
        self._nT = round(T/dt)
        self.Td = Dsp_ratio * self.Ts




        for i in range(1, T_num):
            self.Tx[i, 0] = self.Tx[i - 1, 0] + self.Ts[i - 1, 0]

        ### optimization loop_time interval
        self.dt = dt
        self.t = np.arange(self.dt, self.Tx[T_num - 1, 0] + self.dt, self.dt)
        self.Nsum = len(self.t)

        #### for optimization
        self.comx = np.zeros([self.Nsum, 1])
        self.comvx = np.zeros([self.Nsum, 1])
        self.comax = np.zeros([self.Nsum, 1])
        self.comy = np.zeros([self.Nsum, 1])
        self.comvy = np.zeros([self.Nsum, 1])
        self.comay = np.zeros([self.Nsum, 1])
        self.comz = np.ones([self.Nsum, 1])
        self.comvz = np.zeros([self.Nsum, 1])
        self.comaz = np.zeros([self.Nsum, 1])

        self.px = np.zeros([self.Nsum, 1])
        self.py = np.zeros([self.Nsum, 1])
        self.pz = np.zeros([self.Nsum, 1])
        self.zmpvx = np.zeros([self.Nsum, 1])
        self.zmpvy = np.zeros([self.Nsum, 1])
        self.COMx_is = np.zeros([T_num, 1])
        self.COMx_es = np.zeros([T_num, 1])
        self.COMvx_is = np.zeros([T_num, 1])
        self.COMy_is = np.zeros([T_num, 1])
        self.COMy_es = np.zeros([T_num, 1])
        self.COMvy_is = np.zeros([T_num, 1])

        self.bjxx = 0
        self.bjx1 = 0
        self._t_end_footstep = round((self.Tx[-1,0] - 2 * self.T) / self.dt)
        self.right_support = 0


        # foot state:
        self._Lfootx = np.zeros([self.Nsum, 1])
        self._Lfooty = self._stepwidth[0,0]* np.ones([self.Nsum, 1])
        self._Lfootz = np.zeros([self.Nsum, 1])
        self._Lfootvx = np.zeros([self.Nsum, 1])
        self._Lfootvy = np.zeros([self.Nsum, 1])
        self._Lfootvz = np.zeros([self.Nsum, 1])
        self._Lfootax = np.zeros([self.Nsum, 1])
        self._Lfootay = np.zeros([self.Nsum, 1])
        self._Lfootaz = np.zeros([self.Nsum, 1])

        self._Rfootx = np.zeros([self.Nsum, 1])
        self._Rfooty = -self._stepwidth[0,0]* np.ones([self.Nsum, 1])
        self._Rfootz = np.zeros([self.Nsum, 1])
        self._Rfootvx = np.zeros([self.Nsum, 1])
        self._Rfootvy = np.zeros([self.Nsum, 1])
        self._Rfootvz = np.zeros([self.Nsum, 1])
        self._Rfootax = np.zeros([self.Nsum, 1])
        self._Rfootay = np.zeros([self.Nsum, 1])
        self._Rfootaz = np.zeros([self.Nsum, 1])

        self._ry_left_right = -self._stepwidth[0,0]

    def step_location(self):
        for i in range(1, self._footstepsnumber):
            self._footx_ref[i] = self._footx_ref[i-1] + self._steplength[i-1]
            self._footy_ref[i] = self._footy_ref[i-1] + np.power(-1, i-1) * self._stepwidth[i-1]
            self._footz_ref[i] = self._footz_ref[i-1] + self._stepheight[i-1]

        self._footy_ref[0,0] = -self._stepwidth[0,0]

        print('##################################################')
        print((self._footx_ref).T)
        print((self._footy_ref).T)
        print((self._footz_ref).T)
        print('##################################################')

        return (self.Nsum - 1*self._nT)


    def index_find(self, i, t_array, index_type):
        a = -1
        l = len(t_array)
        if index_type == 0:
            # print("find index")
            for ti in range(0, l - 1):
                # print(t_array[ti, 0])
                # print(t_array[ti+1, 0])
                # print(i * self.dt)
                if (i * self.dt > t_array[ti, 0]) and (i * self.dt <= t_array[ti + 1, 0]):
                    a = ti + 1

                    # print(a)
            return a


    def Ref_com_lip_update(self, i,_hcom):
        # print(i)
        self.bjxx = self.index_find(i,t_array=self.Tx,index_type=0)
        self.bjx1 = self.index_find(i+1, t_array=self.Tx, index_type=0)

        _g = 9.8
        _Wn = np.sqrt(_g / _hcom)
        _Wndt = _Wn * self.dt

        self.comz = _hcom * np.ones([self.Nsum,1])

        self.footx_real = copy.deepcopy(self._footx_ref)
        self.footy_real = copy.deepcopy(self._footy_ref)
        self.footz_real = copy.deepcopy(self._footz_ref)

        for i_period in range(self.bjxx, self.bjxx+2):
            _ki = round(self.Tx[i_period-1, 0] / self.dt) - 1
            if (i_period == 1):
                self.COMx_is[i_period-1,0] = 0
                self.COMy_is[i_period-1,0] = 0
            else:
                self.COMx_is[i_period-1,0] = (self.footx_real[i_period-1,0]+self.footx_real[i_period-2,0]) / 2 - self.footx_real[i_period-1,0]
                self.COMy_is[i_period-1,0] = (self.footy_real[i_period-1,0]+self.footy_real[i_period-2,0]) / 2 - self.footy_real[i_period-1,0]
                
            self.COMx_es[i_period-1,0] = (self.footx_real[i_period,0]+self.footx_real[i_period-1,0]) / 2 - self.footx_real[i_period-1,0]
            self.COMvx_is[i_period-1,0] = (self.COMx_es[i_period-1,0]-self.COMx_is[i_period-1,0] * math.cosh(_Wn * self.Ts[i_period-1,0])) / (1 / _Wn * math.sinh(_Wn * self.Ts[i_period-1,0]))

            self.COMy_es[i_period-1,0] = (self.footy_real[i_period,0]+self.footy_real[i_period-1,0]) / 2 - self.footy_real[i_period-1,0]
            self.COMvy_is[i_period-1,0] = (self.COMy_es[i_period-1,0]-self.COMy_is[i_period-1,0] * math.cosh(_Wn * self.Ts[i_period-1,0])) / (1 / _Wn * math.sinh(_Wn * self.Ts[i_period-1,0]))

        for j in range(1,self._nT+1):
            tj = self.dt * j
            self.px[j+_ki-1,0] =  self.footx_real[i_period-1,0]
            self.py[j+_ki-1,0] =  self.footy_real[i_period-1,0]

            self.comx[j+_ki-1,0] = self.COMx_is[i_period-1,0] * math.cosh(_Wn * tj) + self.COMvx_is[i_period-1,0] * 1 / _Wn * math.sinh(_Wn * tj)+self.px[j+_ki-1,0]
            self.comy[j+_ki-1,0] = self.COMy_is[i_period-1,0] * math.cosh(_Wn * tj) + self.COMvy_is[i_period-1,0] * 1 / _Wn * math.sinh(_Wn * tj)+self.py[j+_ki-1,0]
            self.comvx[j+_ki-1,0] = _Wn * self.COMx_is[i_period-1,0] * math.sinh(_Wn * tj) + self.COMvx_is[i_period-1,0] * math.cosh(_Wn * tj)
            self.comvy[j+_ki-1,0] = _Wn * self.COMy_is[i_period-1,0] * math.sinh(_Wn * tj) + self.COMvy_is[i_period-1,0] * math.cosh(_Wn * tj)
            self.comax[j+_ki-1,0] = _Wn**2 * self.COMx_is[i_period - 1, 0] * math.cosh(_Wn * tj) + _Wn * self.COMvx_is[i_period - 1, 0] * math.sinh(_Wn * tj)
            self.comay[j+_ki-1,0] = _Wn**2 * self.COMy_is[i_period - 1, 0] * math.cosh(_Wn * tj) + _Wn * self.COMvy_is[i_period - 1, 0] * math.sinh(_Wn * tj)
            self.comz[j + _ki - 1,0] = _hcom
            self.comvz[j + _ki - 1, 0] = 0
            self.comaz[j + _ki - 1, 0] = 0

        pos_com = np.zeros([11, 1])
        pos_com[0, 0] = self.comx[i,0]
        pos_com[1, 0] = self.comy[i, 0]
        pos_com[2, 0] = (self.comz[i, 0] - _hcom)
        pos_com[3, 0] = self.comvx[i,0]
        pos_com[4, 0] = self.comvy[i, 0]
        pos_com[5, 0] = self.comvz[i, 0]
        pos_com[6, 0] = self.comax[i,0]
        pos_com[7, 0] = self.comay[i, 0]
        pos_com[8, 0] = self.comaz[i, 0]
        pos_com[9, 0] = self.px[i, 0]
        pos_com[10, 0] = self.py[i, 0]

        return pos_com

    def FootpR(self,j_index):
        rffoot_traj = np.zeros([18,1])
        right_leg_support = 0 ### 2:double support, 1:right_support, 0:left support

        if (j_index >self._t_end_footstep):
            for i_t in range(self.bjx1+1, self._footstepsnumber):
                self._lift_height_ref[i_t-1,0] = 0

        if ((self.bjx1 >= 2) and (j_index <=self._t_end_footstep)):

            if (self.bjx1 % 2 == 0):  # odd: left support
                self.right_support = 0
                self._Lfootx[j_index,0] = self._Lfootx[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                self._Lfooty[j_index,0] = self._Lfooty[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                self._Lfootz[j_index,0] = self._Lfootz[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
    
                self._Lfootx[j_index+1,0] = self._Lfootx[j_index,0]
                self._Lfooty[j_index+1,0] = self._Lfooty[j_index,0]
                self._Lfootz[j_index+1,0] = self._Lfootz[j_index,0]

                if ((j_index + 1 - round(self.Tx[self.bjx1 - 1, 0] / self.dt)) * self.dt <= self.Td[self.bjx1 - 1, 0]):
                    self._Rfootx[j_index,0] = self._Rfootx[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                    self._Rfooty[j_index,0] = self._Rfooty[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                    self._Rfootz[j_index,0] = self._Rfootz[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]

                    self._Rfootx[j_index+1,0] = self._Rfootx[j_index,0]
                    self._Rfooty[j_index+1,0] = self._Rfooty[j_index,0]
                    self._Rfootz[j_index+1,0] = self._Rfootz[j_index,0]

                    self.right_support = 2
                else:
                    t_des = (j_index + 1 - round(self.Tx[self.bjx1 - 1, 0] / self.dt) + 1) * self.dt
                    t_plan = np.zeros([3,1])
                    t_plan[0,0] = t_des - self.dt+0.0005
                    t_plan[1,0] = (self.Td[self.bjx1 - 1, 0] + self.Ts[self.bjx1 - 1, 0]) / 2 + 0.0005
                    t_plan[2,0] = self.Ts[self.bjx1 - 1, 0]  + 0.0005

                    if (abs(t_des - self.Ts[self.bjx1 - 1, 0]) <= 2*self.dt):
                        self._Rfootx[j_index, 0] = self.footx_real[self.bjxx,0]
                        self._Rfooty[j_index, 0] = self.footy_real[self.bjxx,0]
                        self._Rfootz[j_index, 0] = self.footz_real[self.bjxx,0]

                        self._Rfootx[j_index + 1, 0] = self.footx_real[self.bjxx,0]
                        self._Rfooty[j_index + 1, 0] = self.footy_real[self.bjxx,0]
                        self._Rfootz[j_index + 1, 0] = self.footz_real[self.bjxx,0]

                        self.right_support = 2
                    else:
                        AAA_inv = self.solve_AAA_inv2(t_plan)

                        t_a_plan = np.zeros([1,4])
                        t_a_plan[0,0] = np.power(t_des, 3)
                        t_a_plan[0,1] = np.power(t_des, 2)
                        t_a_plan[0,2] = np.power(t_des, 1)
                        t_a_plan[0,3] = 1

                        t_a_planv = np.zeros([1,4])
                        t_a_planv[0,0] = 3 * np.power(t_des, 2)
                        t_a_planv[0,1] = 2 * np.power(t_des, 1)
                        t_a_planv[0,2] = t_des
                        t_a_planv[0,3] = 0

                        ################### Rfootx plan#############################
                        Rfootx_plan = np.zeros([4, 1])
                        Rfootx_plan[0, 0] = self._Rfootx[j_index-1,0]
                        Rfootx_plan[1, 0] = (self.footx_real[self.bjxx-2,0] + self.footx_real[self.bjxx,0])/2
                        Rfootx_plan[2, 0] = self.footx_real[self.bjxx,0]
                        Rfootx_plan[3, 0] = 0

                        Rfootx_co = np.dot(AAA_inv, Rfootx_plan)
                        self._Rfootx[j_index] = np.dot(t_a_plan, Rfootx_co)
                        self._Rfootvx[j_index] = np.dot(t_a_planv, Rfootx_co)

                        ################### Rfooty plan#############################
                        if ((j_index + 1 - round(self.Tx[self.bjx1 - 1,0] / self.dt)) * self.dt < self.Td[self.bjx1 - 1,0] + self.dt):
                            self._ry_left_right = (self.footy_real[self.bjxx,0] + self.footy_real[self.bjxx-2,0]) / 2

                        Rfooty_plan = np.zeros([4, 1])
                        Rfooty_plan[0, 0] = self._Rfooty[j_index-1,0]
                        Rfooty_plan[1, 0] = self._ry_left_right
                        Rfooty_plan[2, 0] = self.footy_real[self.bjxx,0]
                        Rfooty_plan[3, 0] = 0
                        # if (self.bjx1==2):
                        #     print(Rfooty_plan.T)

                        Rfooty_co = np.dot(AAA_inv, Rfooty_plan)
                        self._Rfooty[j_index] = np.dot(t_a_plan, Rfooty_co)
                        self._Rfootvy[j_index] = np.dot(t_a_planv, Rfooty_co)

                        Rfootz_plan = np.zeros([4, 1])
                        Rfootz_plan[0, 0] = self._Rfootz[j_index - 1, 0]
                        Rfootz_plan[1, 0] = max([self.footz_real[self.bjxx-2,0],self.footz_real[self.bjxx,0]])+self._lift_height_ref[self.bjx1-1,0]
                        Rfootz_plan[2, 0] = self.footz_real[self.bjxx,0]
                        Rfootz_plan[3, 0] = 0

                        Rfootz_co = np.dot(AAA_inv, Rfootz_plan)
                        self._Rfootz[j_index] = np.dot(t_a_plan, Rfootz_co)
                        self._Rfootvz[j_index] = np.dot(t_a_planv, Rfootz_co)

                        self._Rfootx[j_index+1,0] = self._Rfootx[j_index,0]+self.dt * self._Rfootvx[j_index,0]
                        self._Rfooty[j_index+1,0] = self._Rfooty[j_index,0]+self.dt * self._Rfootvy[j_index,0]
                        self._Rfootz[j_index+1,0] = self._Rfootz[j_index,0]+self.dt * self._Rfootvz[j_index,0]
            else: ### right support
                self.right_support = 1
                self._Rfootx[j_index, 0] = self._Rfootx[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                self._Rfooty[j_index, 0] = self._Rfooty[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                self._Rfootz[j_index, 0] = self._Rfootz[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]

                self._Rfootx[j_index + 1, 0] = self._Rfootx[j_index, 0]
                self._Rfooty[j_index + 1, 0] = self._Rfooty[j_index, 0]
                self._Rfootz[j_index + 1, 0] = self._Rfootz[j_index, 0]

                if ((j_index + 1 - round(self.Tx[self.bjx1 - 1, 0] / self.dt)) * self.dt <= self.Td[self.bjx1 - 1, 0]):
                    self._Lfootx[j_index, 0] = self._Lfootx[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                    self._Lfooty[j_index, 0] = self._Lfooty[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]
                    self._Lfootz[j_index, 0] = self._Lfootz[round(self.Tx[self.bjx1 - 1, 0] / self.dt) - 1 - 1, 0]

                    self._Lfootx[j_index + 1, 0] = self._Lfootx[j_index, 0]
                    self._Lfooty[j_index + 1, 0] = self._Lfooty[j_index, 0]
                    self._Lfootz[j_index + 1, 0] = self._Lfootz[j_index, 0]
                    self.right_support = 2
                else:
                    t_des = (j_index + 1 - round(self.Tx[self.bjx1 - 1, 0] / self.dt) + 1) * self.dt
                    t_plan = np.zeros([3, 1])
                    t_plan[0, 0] = t_des - self.dt + +0.0005
                    t_plan[1, 0] = (self.Td[self.bjx1 - 1, 0] + self.Ts[self.bjx1 - 1, 0]) / 2 + 0.0005
                    t_plan[2, 0] = self.Ts[self.bjx1 - 1, 0] + 0.0005

                    if (abs(t_des - self.Ts[self.bjx1 - 1, 0]) <= 2*self.dt):
                        self._Lfootx[j_index, 0] = self.footx_real[self.bjxx,0]
                        self._Lfooty[j_index, 0] = self.footy_real[self.bjxx,0]
                        self._Lfootz[j_index, 0] = self.footz_real[self.bjxx,0]

                        self._Lfootx[j_index + 1, 0] = self.footx_real[self.bjxx,0]
                        self._Lfooty[j_index + 1, 0] = self.footy_real[self.bjxx,0]
                        self._Lfootz[j_index + 1, 0] = self.footz_real[self.bjxx,0]

                        self.right_support = 2
                    else:
                        AAA_inv = self.solve_AAA_inv2(t_plan)

                        t_a_plan = np.zeros([1, 4])
                        t_a_plan[0, 0] = np.power(t_des, 3)
                        t_a_plan[0, 1] = np.power(t_des, 2)
                        t_a_plan[0, 2] = np.power(t_des, 1)
                        t_a_plan[0, 3] = 1

                        t_a_planv = np.zeros([1, 4])
                        t_a_planv[0, 0] = 3 * np.power(t_des, 2)
                        t_a_planv[0, 1] = 2 * np.power(t_des, 1)
                        t_a_planv[0, 2] = t_des
                        t_a_planv[0, 3] = 0

                        ################### Lfootx plan#############################
                        Lfootx_plan = np.zeros([4, 1])
                        Lfootx_plan[0, 0] = self._Lfootx[j_index - 1, 0]
                        Lfootx_plan[1, 0] = (self.footx_real[self.bjxx-2,0] + self.footx_real[self.bjxx,0]) / 2
                        Lfootx_plan[2, 0] = self.footx_real[self.bjxx,0]
                        Lfootx_plan[3, 0] = 0

                        Lfootx_co = np.dot(AAA_inv, Lfootx_plan)
                        self._Lfootx[j_index] = np.dot(t_a_plan, Lfootx_co)
                        self._Lfootvx[j_index] = np.dot(t_a_planv, Lfootx_co)

                        ################### Lfooty plan#############################
                        if ((j_index + 1 - round(self.Tx[self.bjx1 - 1, 0] / self.dt)) * self.dt < self.Td[self.bjx1 - 1, 0] + self.dt):
                            self._ry_left_right = (self.footy_real[self.bjxx,0] + self.footy_real[self.bjxx-2,0]) / 2

                        Lfooty_plan = np.zeros([4, 1])
                        Lfooty_plan[0, 0] = self._Lfooty[j_index - 1, 0]
                        Lfooty_plan[1, 0] = self._ry_left_right
                        Lfooty_plan[2, 0] = self.footy_real[self.bjxx,0]
                        Lfooty_plan[3, 0] = 0

                        Lfooty_co = np.dot(AAA_inv, Lfooty_plan)
                        self._Lfooty[j_index] = np.dot(t_a_plan, Lfooty_co)
                        self._Lfootvy[j_index] = np.dot(t_a_planv, Lfooty_co)

                        Lfootz_plan = np.zeros([4, 1])
                        Lfootz_plan[0] = self._Lfootz[j_index - 1, 0]
                        Lfootz_plan[1, 0] = max([self.footz_real[self.bjxx-2,0], self.footz_real[self.bjxx,0]]) + \
                                            self._lift_height_ref[self.bjx1 - 1, 0]
                        Lfootz_plan[2, 0] = self.footz_real[self.bjxx,0]
                        Lfootz_plan[3, 0] = 0

                        Lfootz_co = np.dot(AAA_inv, Lfootz_plan)
                        self._Lfootz[j_index] = np.dot(t_a_plan, Lfootz_co)
                        self._Lfootvz[j_index] = np.dot(t_a_planv, Lfootz_co)

                        self._Lfootx[j_index + 1, 0] = self._Lfootx[j_index, 0] + self.dt * self._Lfootvx[j_index, 0]
                        self._Lfooty[j_index + 1, 0] = self._Lfooty[j_index, 0] + self.dt * self._Lfootvy[j_index, 0]
                        self._Lfootz[j_index + 1, 0] = self._Lfootz[j_index, 0] + self.dt * self._Lfootvz[j_index, 0]
        else:
            if (j_index > self._t_end_footstep):
                self._Rfootx[j_index, 0] = self._Rfootx[j_index - 1, 0]
                self._Rfooty[j_index, 0] = self._Rfooty[j_index - 1, 0]
                self._Rfootz[j_index, 0] = self._Rfootz[j_index - 1, 0]
                self._Lfootx[j_index, 0] = self._Lfootx[j_index - 1, 0]
                self._Lfooty[j_index, 0] = self._Lfooty[j_index - 1, 0]
                self._Lfootz[j_index, 0] = self._Lfootz[j_index - 1, 0]
                self.right_support = 2
            else:
                self._Rfooty[j_index, 0] = -self._stepwidth[0,0]
                self._Lfooty[j_index, 0] = self._stepwidth[0,0]
                self.right_support = 2

        rffoot_traj[0, 0] = self._Rfootx[j_index, 0]
        rffoot_traj[1, 0] = (self._Rfooty[j_index, 0] - self._Rfooty[0, 0])
        rffoot_traj[2, 0] = self._Rfootz[j_index, 0]

        rffoot_traj[3, 0] = self._Lfootx[j_index, 0]
        rffoot_traj[4, 0] = (self._Lfooty[j_index, 0] - self._Lfooty[0, 0])
        rffoot_traj[5, 0] = self._Lfootz[j_index, 0]

        rffoot_traj[6, 0] = self._Rfootvx[j_index, 0]
        rffoot_traj[7, 0] = self._Rfootvy[j_index, 0]
        rffoot_traj[8, 0] = self._Rfootvz[j_index, 0]

        rffoot_traj[9, 0] = self._Lfootvx[j_index, 0]
        rffoot_traj[10, 0] = self._Lfootvy[j_index, 0]
        rffoot_traj[11, 0] = self._Lfootvz[j_index, 0]

        rffoot_traj[12, 0] = self._Rfootax[j_index, 0]
        rffoot_traj[13, 0] = self._Rfootay[j_index, 0]
        rffoot_traj[14, 0] = self._Rfootaz[j_index, 0]

        rffoot_traj[15, 0] = self._Lfootax[j_index, 0]
        rffoot_traj[16, 0] = self._Lfootay[j_index, 0]
        rffoot_traj[17, 0] = self._Lfootaz[j_index, 0]

        right_leg_support = self.right_support


        return rffoot_traj,right_leg_support

    def solve_AAA_inv2(self,t_plan):

        AAA1 = np.zeros([4,4])
        AAA1[0, 0] = np.power(t_plan[0,0], 3)
        AAA1[0, 1] = np.power(t_plan[0,0], 2)
        AAA1[0, 2] = np.power(t_plan[0,0], 1)
        AAA1[0, 3] = 1
        AAA1[1, 0] = np.power(t_plan[1,0], 3)
        AAA1[1, 1] = np.power(t_plan[1,0], 2)
        AAA1[1, 2] = np.power(t_plan[1,0], 1)
        AAA1[1, 3] = 1
        AAA1[2, 0] = np.power(t_plan[2,0], 3)
        AAA1[2, 1] = np.power(t_plan[2,0], 2)
        AAA1[2, 2] = np.power(t_plan[2,0], 1)
        AAA1[2, 3] = 1
        AAA1[3, 0] = 3 * np.power(t_plan[2,0], 2)
        AAA1[3, 1] = 2 * np.power(t_plan[2,0], 1)
        AAA1[3, 2] = np.power(t_plan[2,0], 0)
        AAA1[3, 3] = 0


        AAA1_inv = np.linalg.inv(AAA1)

        return AAA1_inv
