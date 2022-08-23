
# -*- encoding: UTF-8 -*-
import numpy as np
import math
import string

import copy
import sys
import mosek

import time
from functools import wraps

from KMP_class import KMP

# load global variables from external files: not using here
##import global_variables
class NLP:
  def __init__(self, nstep, dt_sample, dt_nlp, hcomx, dsp_ratio, sx, sy, sz, st,lift_height,rleg_traj_refx, lleg_traj_refx, inDim, outDim, kh, lamda, pvFlag, gait_mode):
    ## reference step parameters

    rleg_traj_ref = copy.deepcopy(rleg_traj_refx)
    lleg_traj_ref = copy.deepcopy(lleg_traj_refx)
    row_num = rleg_traj_ref.shape[0]
    col_num = rleg_traj_ref.shape[1]

    self.r_kmp = KMP(rleg_traj_ref, row_num, inDim, outDim, kh, lamda, pvFlag)
    self.l_kmp = KMP(lleg_traj_ref, row_num, inDim, outDim, kh, lamda, pvFlag)



    self.Tn = nstep
    self.hcom = hcomx
    self.g = 9.8

    #### gait generation for pacing and trotting ####
    if(gait_mode<=1):
        self.steplength = sx * np.ones([nstep, 1])
        self.steplength[0, 0] = 0
        self.steplength[1, 0] = 0
        self.steplength[2, 0] = sx / 2
        self.steplength[10, :] = 0
        self.steplength[11, :] = -sx / 2
        self.steplength[12:19, :] = -sx / 2 * np.ones([7, 1])
        xxx = nstep - 17
        self.steplength[17:nstep, :] = np.zeros([xxx, 1])

        self.stepwidth = sy * np.ones([nstep, 1])
        self.stepwidth[0, 0] = self.stepwidth[0, 0] / 2

        self.stepheight = sz * np.ones([nstep, 1])
        self.stepheight[0, 0] = 0
    else:
        self.steplength = sx * np.zeros([nstep, 1])
        self.steplength[0, 0] = 0
        self.steplength[1, 0] = 0


        xxx = nstep - 17
        self.steplength[17:nstep, :] = np.zeros([xxx, 1])

        self.stepwidth = sy * np.ones([nstep, 1])
        self.stepwidth[0, 0] = self.stepwidth[0, 0] / 2

        self.stepheight = sz * np.ones([nstep, 1])
        self.stepheight[0, 0] = 0


    self.liftheight = lift_height

    ## reference steplengh and self.stepwidth
    self.Lxx_ref = copy.deepcopy(self.steplength)
    self.Lyy_ref = copy.deepcopy(self.stepwidth)

    for i in range(0, nstep):
        self.Lyy_ref[i, 0] = (-1) ** (i) * self.stepwidth[i, 0]

    # reference location (update online):
    self.footx_ref = np.zeros([nstep, 1])
    self.footy_ref = np.zeros([nstep, 1])
    self.footz_ref = np.zeros([nstep, 1])
    for i in range(1, nstep):  # %%% singular period ===> right support
        self.footx_ref[i, 0] = self.footx_ref[i - 1, 0] + self.steplength[i - 1, 0]
        self.footy_ref[i, 0] = self.footy_ref[i - 1, 0] + pow(-1, i - 1) * self.stepwidth[i - 1, 0]
        self.footz_ref[i, 0] = self.footz_ref[i - 1, 0] + self.stepheight[i - 1, 0]

        # reference location (offline decided)
    self.footx_offline = copy.deepcopy(self.footx_ref)
    self.footy_offline = copy.deepcopy(self.footy_ref)
    self.footz_offline = copy.deepcopy(self.footz_ref)
    print("footy_offline:",(self.footy_offline).T)
    print("Sy:", (self.stepwidth).T)

    ### reference walking period
    self.Ts = st * np.ones([nstep, 1])
    self.Tx = np.zeros([nstep, 1])
    self.dsp = dsp_ratio

    for i in range(1, nstep):
        self.Tx[i, 0] = self.Tx[i - 1, 0] + self.Ts[i - 1, 0]

    ## sampling time
    self.dtx = dt_sample
    self.tx = np.arange(self.dtx, self.Tx[nstep - 1, 0] + self.dtx, self.dtx)
    self.Nsum1 = len(self.tx)

    ### optimization loop_time interval
    self.dt = dt_nlp
    self.t = np.arange(self.dt, self.Tx[nstep - 1, 0] + self.dt, self.dt)
    self.Nsum = len(self.t)

    self.Lxx_ref_real = np.zeros([self.Nsum, 1])
    self.Lyy_ref_real = np.zeros([self.Nsum, 1])
    self.Ts_ref_real = np.zeros([self.Nsum, 1])
    #### for optimization
    self.comx = np.zeros([self.Nsum, 1])
    self.comvx = np.zeros([self.Nsum, 1])
    self.comax = np.zeros([self.Nsum, 1])
    self.comy = np.zeros([self.Nsum, 1])
    self.comvy = np.zeros([self.Nsum, 1])
    self.comay = np.zeros([self.Nsum, 1])
    self.comz = hcomx * np.ones([self.Nsum, 1])
    self.comvz = np.zeros([self.Nsum, 1])
    self.comaz = np.zeros([self.Nsum, 1])
    ### for lower lever control
    self.comx1 = np.zeros([self.Nsum1, 1])
    self.comvx1 = np.zeros([self.Nsum1, 1])
    self.comax1 = np.zeros([self.Nsum1, 1])
    self.comy1 = np.zeros([self.Nsum1, 1])
    self.comvy1 = np.zeros([self.Nsum1, 1])
    self.comay1 = np.zeros([self.Nsum1, 1])
    self.comz1 = hcomx * np.ones([self.Nsum1, 1])
    self.comvz1 = np.zeros([self.Nsum1, 1])
    self.comaz1 = np.zeros([self.Nsum1, 1])

    self.px = np.zeros([self.Nsum, 1])
    self.py = np.zeros([self.Nsum, 1])
    self.pz = np.zeros([self.Nsum, 1])
    self.cpx = np.zeros([self.Nsum, 1])
    self.cpy = np.zeros([self.Nsum, 1])
    self.cpx_T = np.zeros([self.Nsum, 1])
    self.cpy_T = np.zeros([self.Nsum, 1])
    self.zmpvx = np.zeros([self.Nsum, 1])
    self.zmpvy = np.zeros([self.Nsum, 1])
    self.COMx_is = np.zeros([nstep, 1])
    self.COMx_es = np.zeros([nstep, 1])
    self.COMvx_is = np.zeros([nstep, 1])
    self.COMy_is = np.zeros([nstep, 1])
    self.COMy_es = np.zeros([nstep, 1])
    self.COMvy_is = np.zeros([nstep, 1])

    # real-state: state feedback,given by state measurement and state estimation
    self.comx_feed = np.zeros([self.Nsum, 1])
    self.comvx_feed = np.zeros([self.Nsum, 1])
    self.comax_feed = np.zeros([self.Nsum, 1])
    self.comy_feed = np.zeros([self.Nsum, 1])
    self.comvy_feed = np.zeros([self.Nsum, 1])
    self.comay_feed = np.zeros([self.Nsum, 1])
    self.comz_feed = np.zeros([self.Nsum, 1])
    self.comvz_feed = np.zeros([self.Nsum, 1])
    self.comaz_feed = np.zeros([self.Nsum, 1])

    # optimal variables:
    self.Vari_ini = np.zeros([4, self.Nsum])  # Lxx,Lyy,Tr1,Tr2,Lxx1,Lyy1,Tr11,Tr21;

    # test time consumption:
    self.tcpu = np.zeros([self.Nsum - 32, 1])

    self.ini = 1

    self.comvx_endref = np.zeros([1, 1])
    self.comvy_endref = np.zeros([1, 1])

    self.rfoot_kmp = np.zeros([3, 1])
    self.rfoot_kmp[1, 0] = -abs(self.footy_ref[3, 0])
    self.lfoot_kmp = np.zeros([3, 1])
    self.lfoot_kmp[1, 0] = abs(self.footy_ref[3, 0])
    self.rfootv_kmp = np.zeros([3, 1])
    self.lfootv_kmp = np.zeros([3, 1])

    print("NLP initialization!!!!!!!!!!!!!")

  def index_find(self, i, t_array, index_type):
      a = -1
      l = len(t_array)
      if index_type == 0:
          for ti in range(0, l - 1):
              if (i * self.dt > t_array[ti, 0]) and (i * self.dt <= t_array[ti + 1, 0]):
                  a = ti + 1
          return a


  def fun_sdr_quadratic_constraint(self, H2_q_xy, H1_q_xy, Nh, nt_rea):
      cin_quad = np.zeros([Nh, nt_rea, nt_rea])
      for jj in range(0, Nh):
          cin_quad[jj, 0:4, 0:4] = H2_q_xy[jj, :, :]
          cin_quad[jj, 0:4, 4] = 0.5 * np.transpose(H1_q_xy[jj, :])
          cin_quad[jj, 4, 0:4] = 0.5 * H1_q_xy[jj, :]
          cin_quad[jj, 4, 4] = 0

      return cin_quad


  def fun_sdr_affine_constraint(self, A_q1, nt_rea):
      a = A_q1.shape[0]
      b = A_q1.shape[1]
      cin_aff = np.zeros([a, nt_rea, nt_rea])
      xx = np.zeros([b, b])
      for jj in range(0, a):
          cin_aff[jj, 0:4, 4] = 0.5 * np.transpose(A_q1[jj, :])
          cin_aff[jj, 4, 0:4] = 0.5 * A_q1[jj, :]
          cin_aff[jj, 4, 4] = 0

      return cin_aff, a


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


  # @fn_timer
  def nlp_gait(self, i,gait_mode):
      ### objective coefficients
      aax = 1000000
      aay = 10000
      aaxv = 100
      aayv = 1000
      bbx = 10000000
      bby = 10000000
      rr1 = 1000000
      rr2 = 1000000
      if(gait_mode==0):
          aax = 1000000
          aay = 10000
          aaxv = 100
          aayv = 1000
          bbx = 10000000
          bby = 10000000
          rr1 = 1000000
          rr2 = 1000000

      inf = 0.0
      ## LIPM model parameters
      M_f = 12

      Wn = np.sqrt(self.g / self.hcom)
      ##print(Wn)

      # ZMP constraints
      px_max = 0.07
      px_min = -0.03
      py_max = 0.05
      py_min = -0.05

      ##DCM constraints
      cpx_max = 0.07
      cpx_min = -0.03
      cpy_max = 0.05
      cpy_min = -0.05

      # constraints- bounddaries parameters initialization
      ## step time variation
      t_min = 0.4
      t_max = 1.2

      # swing foot maximal velocity
      footx_vmax = 1.0  # %%%%2*L_max/t_min = 0.2*2/0.4=1.0
      footx_vmin = -1.0  # %%%%2*-L_max/t_min = -0.2*2/0.4=1.0
      footy_vmax = 0.75  # %%%% 2*det_W_max/t_min = 2*0.15/0.4=0.75
      footy_vmin = -0.75

      # CoM acceleration velocity:
      comax_max = 4
      comax_min = -4
      comay_max = 8
      comay_min = -8

      # %%%external force

      FX = 300
      FY = 200
      t_last = 0.1
      det_xa = (FX + 0.1) / M_f
      det_ya = (FY + 0.1) / M_f
      det_xv = det_xa * t_last
      det_yv = det_ya * t_last
      det_xp = det_xv ** 2 / (2.0 * det_xa)
      det_yp = det_yv ** 2 / (2.0 * det_ya)

      time_start = time.time()

      ############################# optimization preparation####################
      ###################################################################
      k = self.index_find(i, self.Tx, 0)

      self.px[i, 0] = self.footx_ref[k - 1, 0]
      self.zmpvx[i, :] = 0
      self.py[i, 0] = self.footy_ref[k - 1, 0]
      self.zmpvy[i, :] = 0

      ki = np.round(self.Tx[k - 1, 0] / self.dt)
      k_yu = (int)(i - ki)

      TK = self.Ts[k - 1, 0] - k_yu * self.dt
      # print('k_yu:\n', TK)

      ### step location reference
      Lxx_refx = self.Lxx_ref[k - 1, 0]
      Lyy_refy = self.Lyy_ref[k - 1, 0]  # tracking the step length and width

      tr1_ref = np.cosh(Wn * TK)
      tr2_ref = np.sinh(Wn * TK)

      vari_ini = np.zeros([4, 1])
      vari_ini[0, 0] = Lxx_refx
      vari_ini[1, 0] = Lyy_refy
      vari_ini[2, 0] = tr1_ref
      vari_ini[3, 0] = tr2_ref

      ## feasibility constraints
      ## step time constraints
      if (t_min - k_yu * self.dt >= 0.005):
          tr1_min = np.cosh(Wn * (t_min - k_yu * self.dt))
          tr2_min = np.sinh(Wn * (t_min - k_yu * self.dt))
      else:
          tr1_min = np.cosh(Wn * (0.005))
          tr2_min = np.sinh(Wn * (0.005))

      tr1_max = np.cosh(Wn * (t_max - k_yu * self.dt))
      tr2_max = np.sinh(Wn * (t_max - k_yu * self.dt))

      ##foot location constraints
      Footx_max1 = 0.2
      Footx_min1 = -0.1

      if ((k % 2) == 1):  ##singular number
          Footy_max1 = 0.28
          if (k == 1):
              Footy_min1 = 0.1
          else:
              Footy_min1 = 0.18
      else:
          Footy_max1 = -0.18
          Footy_min1 = -0.28

      ### DCM constraints
      if (t_min - k_yu * self.dt >= 0.005):
          cpx_max = Footx_max1 / (np.exp(Wn * t_min) - 1)
          cpx_min = 30 * Footx_min1 / (np.exp(Wn * t_min) - 1)
      else:
          cpx_max = Footx_max1 / (np.exp(Wn * (0.005 + k_yu * self.dt)) - 1)
          cpx_min = 30 * Footx_min1 / (np.exp(Wn * (0.005 + k_yu * self.dt)) - 1)

      if ((k % 2) == 1):
          lp = 0.145
          W_max = Footy_max1 - lp
          W_min = Footy_min1 - lp
          if (t_min - k_yu * self.dt >= 0.005):
              cpy_max = - lp / (np.exp(Wn * t_max) + 1) + W_max / (np.exp(2 * Wn * t_min) - 1)
              cpy_min = - lp / (np.exp(Wn * t_min) + 1) + W_min / (np.exp(2 * Wn * t_min) - 1)
          else:
              cpy_max = - lp / (np.exp(Wn * t_max) + 1) + W_max / (np.exp(2 * Wn * (0.005 + k_yu * self.dt)) - 1)
              cpy_min = - lp / (np.exp(Wn * (0.005 + k_yu * self.dt)) + 1) + W_min / (np.exp(2 * Wn * (0.005 + k_yu * self.dt)) - 1)
      else:
          lp = -0.145
          W_max = Footy_max1 - lp
          W_min = Footy_min1 - lp
          if (t_min - k_yu * self.dt >= 0.005):
              cpy_max = - lp / (np.exp(Wn * t_min) + 1) + W_max / (np.exp(2 * Wn * t_min) - 1)
              cpy_min = - lp / (np.exp(Wn * t_max) + 1) + W_min / (np.exp(2 * Wn * t_min) - 1)
          else:
              cpy_max = - lp / (np.exp(Wn * (0.005 + k_yu * self.dt)) + 1) + W_max / (np.exp(2 * Wn * (0.005 + k_yu * self.dt)) - 1)
              cpy_min = - lp / (np.exp(Wn * t_max) + 1) + W_min / (np.exp(2 * Wn * (0.005 + k_yu * self.dt)) - 1)

      S1 = np.zeros([1, 4])
      S1[0, 0] = 1
      S2 = np.zeros([1, 4])
      S2[0, 1] = 1
      S3 = np.zeros([1, 4])
      S3[0, 2] = 1
      S4 = np.zeros([1, 4])
      S4[0, 3] = 1

      if (i == 1):
          self.COMx_is[k - 1, 0] = self.comx_feed[i - 1, 0] - self.footx_ref[k - 1, 0]
          self.COMx_es[k - 1, 0] = np.dot(S1, vari_ini) / 2
          self.COMvx_is[k - 1, 0] = (self.COMx_es[k - 1, 0] - self.COMx_is[k - 1, 0] * np.dot(S3, vari_ini)) / (
                      1 / Wn * np.dot(S4, vari_ini))
          self.COMy_is[k - 1, 0] = self.comy_feed[i - 1, 0] - self.footy_ref[k - 1, 0]
          self.COMy_es[k - 1, 0] = np.dot(S2, vari_ini) / 2
          self.COMvy_is[k - 1, 0] = (self.COMy_es[k - 1, 0] - self.COMy_is[k - 1, 0] * np.dot(S3, vari_ini)) / (
                      1 / Wn * np.dot(S4, vari_ini))
          self.comvx_endref = Wn * self.COMx_is[k - 1, 0] * np.dot(S4, vari_ini) + self.COMvx_is[k - 1, 0] * np.dot(S3, vari_ini)
          self.comvy_endref = Wn * self.COMy_is[k - 1, 0] * np.dot(S4, vari_ini) + self.COMvy_is[k - 1, 0] * np.dot(S3, vari_ini)

      ### step time & step location optimization####################################
      ###objective function
      AxO = self.comx_feed[i - 1, 0] - self.footx_ref[k - 1, 0]
      BxO = self.comvx_feed[i - 1, 0] / Wn
      Cx = -1.0 / 2 * Lxx_refx
      Axv = Wn * BxO
      Bxv = Wn * AxO
      Cxv = -self.comvx_endref[0, 0]
      AyO = self.comy_feed[i - 1, 0] - self.footy_ref[k - 1, 0]
      ByO = self.comvy_feed[i - 1, 0] / Wn
      Cy = -1.0 / 2 * Lyy_refy
      Ayv = Wn * ByO
      Byv = Wn * AyO
      Cyv = -self.comvy_endref[0, 0]
      Q_goal1 = np.zeros([4, 4])
      Q_goal1[0, 0] = bbx
      Q_goal1[1, 1] = bby
      Q_goal1[2, 2] = rr1 + aax * (AxO ** 2) + aay * AyO ** 2 + aaxv * Axv ** 2 + aayv * Ayv ** 2
      Q_goal1[2, 3] = aax * AxO * BxO + aay * AyO * ByO + aaxv * Axv * Bxv + aayv * Ayv * Byv
      Q_goal1[3, 2] = Q_goal1[2, 3]
      Q_goal1[3, 3] = rr2 + aax * BxO ** 2 + aay * ByO ** 2 + aaxv * Bxv ** 2 + aayv * Byv ** 2
      Q_goal = (Q_goal1 + np.transpose(Q_goal1)) / 2

      q_goal = np.zeros([4, 1])
      q_goal[0, 0] = 2 * (-bbx) * Lxx_refx
      q_goal[1, 0] = 2 * (-bby) * Lyy_refy
      q_goal[2, 0] = 2 * (-rr1 * tr1_ref + aax * AxO * Cx + aay * AyO * Cy + aaxv * Axv * Cxv + aayv * Ayv * Cyv)
      q_goal[3, 0] = 2 * (-rr2 * tr2_ref + aax * BxO * Cx + aay * ByO * Cy + aaxv * Bxv * Cxv + aayv * Byv * Cyv)
      ################################################################
      ###constraints
      ### quadratic tr1 & tr2: equation constraints:=== transform into inequality constraints
      trx12_up2 = np.dot(np.transpose(S3), S3) - np.dot(np.transpose(S4), S4)
      trx12_up1 = np.zeros([1, 4])
      det_trx12_up = 1
      trx12_lp2 = -(trx12_up2)
      trx12_lp1 = np.zeros([1, 4])
      det_trx12_1p = -1

      # remaining time constraints: inequality constraints
      trx1_up = S3
      det_trx1_up = -(-tr1_max)

      trx1_lp = -S3
      det_trx1_lp = -(tr1_min)

      trx2_up = S4
      det_trx2_up = -(-tr2_max)

      trx2_lp = -S4
      det_trx2_lp = -(tr2_min)

      trx = np.zeros([4, 4])
      trx[0] = trx1_up
      trx[1] = trx1_lp
      trx[2] = trx2_up
      trx[3] = trx2_lp

      det_trx = np.zeros([4, 1])
      det_trx[0] = det_trx1_up
      det_trx[1] = det_trx1_lp
      det_trx[2] = det_trx2_up
      det_trx[3] = det_trx2_lp

      ### step location constraints
      h_lx_up = S1
      det_h_lx_up = -(-Footx_max1)

      h_lx_lp = -S1
      det_h_lx_lp = -(Footx_min1)

      h_ly_up = S2
      det_h_ly_up = -(-Footy_max1)

      h_ly_lp = -S2
      det_h_ly_lp = -(Footy_min1)

      h_lx_upx = np.zeros([4, 4])
      h_lx_upx[0] = h_lx_up
      h_lx_upx[1] = h_lx_lp
      h_lx_upx[2] = h_ly_up
      h_lx_upx[3] = h_ly_lp

      det_h_lx_upx = np.zeros([4, 1])
      det_h_lx_upx[0] = det_h_lx_up
      det_h_lx_upx[1] = det_h_lx_lp
      det_h_lx_upx[2] = det_h_ly_up
      det_h_lx_upx[3] = det_h_ly_lp

      ### swing foot velocity constraints
      if (k_yu == 0):
          h_lvx_up = np.zeros([1, 4])
          det_h_lvx_up = 0.0001

          h_lvx_lp = np.zeros([1, 4])
          det_h_lvx_lp = 0.0001

          h_lvy_up = np.zeros([1, 4])
          det_h_lvy_up = 0.0001

          h_lvy_lp = np.zeros([1, 4])
          det_h_lvy_lp = 0.0001
      else:
          h_lvx_up = S1
          det_h_lvx_up = -(- self.Lxx_ref[k - 1, 0] - footx_vmax * self.dt)

          h_lvx_lp = -S1
          det_h_lvx_lp = (-self.Lxx_ref[k - 1, 0] - footx_vmin * self.dt)

          h_lvy_up = S2
          det_h_lvy_up = -(- self.Lyy_ref[k - 1, 0] - footy_vmax * self.dt)

          h_lvy_lp = -S2
          det_h_lvy_lp = (-self.Lyy_ref[k - 1, 0] - footy_vmin * self.dt)

      h_lvx_upx = np.zeros([4, 4])
      h_lvx_upx[0] = h_lvx_up
      h_lvx_upx[1] = h_lvx_lp
      h_lvx_upx[2] = h_lvy_up
      h_lvx_upx[3] = h_lvy_lp

      det_h_lvx_upx = np.zeros([4, 1])
      det_h_lvx_upx[0] = det_h_lvx_up
      det_h_lvx_upx[1] = det_h_lvx_lp
      det_h_lvx_upx[2] = det_h_lvy_up
      det_h_lvx_upx[3] = det_h_lvy_lp

      ###CoM accelearation boundary
      AA = Wn * np.sinh(Wn * self.dt)
      CCx = self.comx_feed[i - 1, 0] - self.footx_ref[k - 1, 0]
      BBx = Wn ** 2 * CCx * np.cosh(Wn * self.dt)
      CCy = self.comy_feed[i - 1, 0] - self.footy_ref[k - 1, 0]
      BBy = Wn ** 2 * CCy * np.cosh(Wn * self.dt)

      AA1x = AA * Wn
      AA2x = -2 * AA * CCx * Wn
      AA3x = 2 * BBx
      AA1y = AA * Wn
      AA2y = -2 * AA * CCy * Wn
      AA3y = 2 * BBy

      CoM_lax_up = AA1x * S1 + AA2x * S3 + (AA3x - 2 * comax_max) * S4
      det_CoM_lax_up = 0

      CoM_lax_lp = -AA1x * S1 - AA2x * S3 - (AA3x - 2 * comax_min) * S4
      det_CoM_lax_lp = 0

      CoM_lay_up = AA1y * S2 + AA2y * S3 + (AA3y - 2 * comay_max) * S4
      det_CoM_lay_up = 0

      CoM_lay_lp = -AA1y * S2 - AA2y * S3 - (AA3y - 2 * comay_min) * S4
      det_CoM_lay_lp = 0

      CoM_lax_upx = np.zeros([4, 4])
      CoM_lax_upx[0] = CoM_lax_up
      CoM_lax_upx[1] = CoM_lax_lp
      CoM_lax_upx[2] = CoM_lay_up
      CoM_lax_upx[3] = CoM_lay_lp

      det_CoM_lax_upx = np.zeros([4, 1])
      det_CoM_lax_upx[0] = det_CoM_lax_up
      det_CoM_lax_upx[1] = det_CoM_lax_lp
      det_CoM_lax_upx[2] = det_CoM_lay_up
      det_CoM_lax_upx[3] = det_CoM_lay_lp

      ### CoM velocity_inremental boundary
      VAA = np.cosh(Wn * self.dt)
      VCCx = self.comx_feed[i - 1, 0] - self.footx_ref[k - 1, 0]
      VBBx = Wn * VCCx * np.sinh(Wn * self.dt)
      VCCy = self.comy_feed[i - 1, 0] - self.footy_ref[k - 1, 0]
      VBBy = Wn * VCCy * np.sinh(Wn * self.dt)

      VAA1x = VAA * Wn
      VAA2x = -2 * VAA * VCCx * Wn
      VAA3x = 2 * VBBx - 2 * self.comvx_feed[i - 1, 0]
      VAA1y = VAA * Wn
      VAA2y = -2 * VAA * VCCy * Wn
      VAA3y = 2 * VBBy - 2 * self.comvy_feed[i - 1, 0]

      CoM_lvx_up = VAA1x * S1 + VAA2x * S3 + (VAA3x - 2 * comax_max * self.dt) * S4
      det_CoM_lvx_up = 0

      CoM_lvx_lp = -VAA1x * S1 - VAA2x * S3 - (VAA3x - 2 * comax_min * self.dt) * S4
      det_CoM_lvx_lp = 0

      CoM_lvy_up = VAA1y * S2 + VAA2y * S3 + (VAA3y - 2 * comay_max * self.dt) * S4
      det_CoM_lvy_up = 0

      CoM_lvy_lp = -VAA1y * S2 - VAA2y * S3 - (VAA3y - 2 * comay_min * self.dt) * S4
      det_CoM_lvy_lp = 0

      CoM_lvx_upx = np.zeros([4, 4])
      CoM_lvx_upx[0] = CoM_lvx_up
      CoM_lvx_upx[1] = CoM_lvx_lp
      CoM_lvx_upx[2] = CoM_lvy_up
      CoM_lvx_upx[3] = CoM_lvy_lp

      det_CoM_lvx_upx = np.zeros([4, 1])
      det_CoM_lvx_upx[0] = det_CoM_lvx_up
      det_CoM_lvx_upx[1] = det_CoM_lvx_lp
      det_CoM_lvx_upx[2] = det_CoM_lvy_up
      det_CoM_lvx_upx[3] = det_CoM_lvy_lp

      # CoM initial velocity_ boundary
      VAA1x = Wn
      VAA2x = -2 * VCCx * Wn
      VAA3x = -2 * self.comvx_feed[i - 1, 0]
      VAA1y = Wn
      VAA2y = -2 * VCCy * Wn
      VAA3y = -2 * self.comvy_feed[i - 1, 0]

      CoM_vx_up = VAA1x * S1 + VAA2x * S3 + (VAA3x - 2 * comax_max * self.dt) * S4
      det_CoM_vx_up = 0

      CoM_vx_lp = -VAA1x * S1 - VAA2x * S3 - (VAA3x - 2 * comax_min * self.dt) * S4
      det_CoM_vx_lp = 0

      CoM_vy_up = VAA1y * S2 + VAA2y * S3 + (VAA3y - 2 * comay_max * self.dt) * S4
      det_CoM_vy_up = 0

      CoM_vy_lp = -VAA1y * S2 - VAA2y * S3 - (VAA3y - 2 * comay_min * self.dt) * S4
      det_CoM_vy_lp = 0

      CoM_vx_upx = np.zeros([4, 4])
      CoM_vx_upx[0] = CoM_vx_up
      CoM_vx_upx[1] = CoM_vx_lp
      CoM_vx_upx[2] = CoM_vy_up
      CoM_vx_upx[3] = CoM_vy_lp

      det_CoM_vx_upx = np.zeros([4, 1])
      det_CoM_vx_upx[0] = det_CoM_vx_up
      det_CoM_vx_upx[1] = det_CoM_vx_lp
      det_CoM_vx_upx[2] = det_CoM_vy_up
      det_CoM_vx_upx[3] = det_CoM_vy_lp

      # quadratic constraints == DCM constraints
      cp_x_up2 = 1.0 / 2 * np.dot(S1.transpose(), (S3 - S4))
      cp_x_up1 = -cpx_max * S4
      det_cp_x_up = -(-VCCx)

      cp_x_lp2 = -1.0 / 2 * np.dot(S1.transpose(), (S3 - S4))
      cp_x_lp1 = +cpx_min * S4
      det_cp_x_lp = (-VCCx)

      cp_y_up2 = 1.0 / 2 * np.dot(S2.transpose(), (S3 - S4))
      cp_y_up1 = -cpy_max * S4
      det_cp_y_up = -(-VCCy)

      cp_y_lp2 = -1.0 / 2 * np.dot(S2.transpose(), (S3 - S4))
      cp_y_lp1 = cpy_min * S4
      det_cp_y_lp = (-VCCy)

      ##### quadratic programming: A_q1 * X <= b_q1
      A_q1 = np.zeros([24, 4])
      b_q1 = np.zeros([24, 1])
      A_q1[0:4, :] = trx
      A_q1[4:8, :] = h_lx_upx
      A_q1[8:12, :] = h_lvx_upx
      A_q1[12:16, :] = CoM_lax_upx
      A_q1[16:20, :] = CoM_lvx_upx
      A_q1[20:24, :] = CoM_vx_upx

      b_q1[0:4, :] = det_trx
      b_q1[4:8, :] = det_h_lx_upx
      b_q1[8:12, :] = det_h_lvx_upx
      b_q1[12:16, :] = det_CoM_lax_upx
      b_q1[16:20, :] = det_CoM_lvx_upx
      b_q1[20:24, :] = det_CoM_vx_upx

      ###Semidefinite Relaxation
      nt_rea = 5
      Nh = 2
      H2_q_xy = np.zeros([Nh, 4, 4])  ## note that numpy 3d_array: first dim is the number of 2d-array
      H1_q_xy = np.zeros([Nh, 4])
      F_quadratci = np.zeros([Nh, 1])

      H2_q_xy[0, :, :] = trx12_up2
      H1_q_xy[0, :] = trx12_up1
      F_quadratci[0, :] = det_trx12_up
      H2_q_xy[1, :, :] = trx12_lp2
      H1_q_xy[1, :] = trx12_lp1
      F_quadratci[1, :] = det_trx12_1p
      # H2_q_xy[2, :, :] = cp_x_up2
      # H1_q_xy[2, :] = cp_x_up1
      # F_quadratci[2, :] = det_cp_x_up
      # H2_q_xy[3, :, :] = cp_x_lp2
      # H1_q_xy[3, :] = cp_x_lp1
      # F_quadratci[3, :] = det_cp_x_lp
      # H2_q_xy[4, :, :] = cp_y_up2
      # H1_q_xy[4, :] = cp_y_up1
      # F_quadratci[4, :] = det_cp_y_up
      # H2_q_xy[5, :, :] = cp_y_lp2
      # H1_q_xy[5, :] = cp_y_lp1
      # F_quadratci[5, :] = det_cp_y_lp


      cin_quad = self.fun_sdr_quadratic_constraint(H2_q_xy, H1_q_xy, Nh, nt_rea)

      cin_aff, a_cin_aff = self.fun_sdr_affine_constraint(A_q1, nt_rea)

      W = np.zeros([nt_rea, nt_rea])
      W[0:4, 0:4] = Q_goal
      W[0:4, 4] = 0.5 * q_goal[:, 0]
      W[4, 0:4] = 0.5 * np.transpose(q_goal[:, 0])

      p4_hat = np.zeros([nt_rea, nt_rea])
      p4_hat[nt_rea - 1, nt_rea - 1] = 1

      if (TK >= 0.15 * self.Ts[k - 1, 0]):
          # print("Solving the SD relaxation of the QCQP problem...")
          ### mosek
          # Make mosek environment
          with mosek.Env() as env:

              # Create a task object and attach log stream printer
              with env.Task(0, 0) as task:
                  #### objective_coefficient
                  barci, barcj, barcval = self.sparse_matrix_express(W)

                  numcon = 1 + 2 + 24
                  BARVARDIM = [5]
                  #
                  # Append 'numcon' empty constraints.
                  task.appendcons(numcon)
                  #
                  # Append matrix variables of sizes in 'BARVARDIM'.
                  task.appendbarvars(BARVARDIM)

                  symc = task.appendsparsesymmat(BARVARDIM[0],
                                                 barci,
                                                 barcj,
                                                 barcval)

                  task.putbarcj(0, [symc], [1.0])

                  ji_list = []
                  bkc_list = []
                  blc_list = []
                  buc_list = []

                  for ji in range(numcon):
                      # Set the bounds on constraints.
                      # blc[i] <= constraint_i <= buc[i]
                      ji_list.append(ji)

                      if (ji == 0):  #### equality constraints
                          bkc_list.append(mosek.boundkey.fx)
                          blc_list.append(1)
                          buc_list.append(1)

                          barai, baraj, baraval = self.sparse_matrix_express(p4_hat)
                      else:
                          ###### quddratic constraints-affine
                          if (ji <= 2):
                              bkc_list.append(mosek.boundkey.up)
                              blc_list.append(-1000000)
                              buc_list.append(F_quadratci[ji - 1, :])

                              barai, baraj, baraval = self.sparse_matrix_express(cin_quad[ji - 1, :, :])
                          else:
                              bkc_list.append(mosek.boundkey.up)
                              blc_list.append(-1000000)
                              buc_list.append(b_q1[ji - 3, :])

                              barai, baraj, baraval = self.sparse_matrix_express(cin_aff[ji - 3, :, :])

                      #### constraint_coefficient
                      syma0 = task.appendsparsesymmat(BARVARDIM[0],
                                                      barai,
                                                      baraj,
                                                      baraval)

                      task.putbaraij(ji, 0, [syma0], [1.0])

                  task.putconboundlist(ji_list, bkc_list, blc_list, buc_list)

                  # Input the objective sense (minimize/maximize)
                  task.putobjsense(mosek.objsense.minimize)

                  # Solve the problem and print summary
                  task.optimize()

                  # Get status information about the solution
                  solsta = task.getsolsta(mosek.soltype.itr)

                  if (solsta == mosek.solsta.optimal):
                      lenbarvar = BARVARDIM[0] * (BARVARDIM[0] + 1) / 2
                      barx = [0.] * int(lenbarvar)
                      task.getbarxj(mosek.soltype.itr, 0, barx)

                      vari_ini[0, 0] = barx[4]
                      vari_ini[1, 0] = barx[8]
                      vari_ini[2, 0] = barx[11]
                      vari_ini[3, 0] = barx[13]

                      X_vacc_k = copy.deepcopy(vari_ini)
                      X_vacc_k[0, 0] = self.footx_ref[k, 0] - self.footx_ref[k - 1, 0]
                      X_vacc_k[1, 0] = self.footy_ref[k, 0] - self.footy_ref[k - 1, 0]
                  else:
                      print('No optimal solutin using mosek!!!\n')
                      X_vacc_k = copy.deepcopy(vari_ini)
                      X_vacc_k[0, 0] = self.footx_ref[k, 0] - self.footx_ref[k - 1, 0]
                      X_vacc_k[1, 0] = self.footy_ref[k, 0] - self.footy_ref[k - 1, 0]
      else:
          X_vacc_k = copy.deepcopy(vari_ini)
          X_vacc_k[0, 0] = self.footx_ref[k, 0] - self.footx_ref[k - 1, 0]
          X_vacc_k[1, 0] = self.footy_ref[k, 0] - self.footy_ref[k - 1, 0]

      # print('Generated X_vacc_k:\n',X_vacc_k)

      vari_ini = copy.deepcopy(X_vacc_k)

      self.Vari_ini[:, i] = vari_ini[:, 0]

      ####################################################################
      ###%%%%%%%%%%%%%%%%%%%%% post-processiong###########################
      ##update the optimal parameter in the real-time
      self.Lxx_ref[k - 1, 0] = np.dot(S1, vari_ini)
      self.Lyy_ref[k - 1, 0] = np.dot(S2, vari_ini)
      self.Ts[k - 1, 0] = k_yu * self.dt + np.log(np.dot((S3 + S4), vari_ini)) / Wn

      self.Lxx_ref_real[i, 0] = self.Lxx_ref[k - 1, 0]
      self.Lyy_ref_real[i, 0] = self.Lyy_ref[k - 1, 0]
      self.Ts_ref_real[i, 0] = self.Ts[k - 1, 0]

      self.COMx_is[k - 1, 0] = self.comx_feed[i - 1, 0] - self.footx_ref[k - 1, 0]
      self.COMx_es[k - 1, 0] = self.Lxx_ref[k - 1, 0] / 2
      self.COMvx_is[k - 1, 0] = (self.COMx_es[k - 1, 0] - self.COMx_is[k - 1, 0] * np.dot(S3, vari_ini)) / (
                  1 / Wn * np.dot(S4, vari_ini))
      self.COMy_is[k - 1, 0] = self.comy_feed[i - 1, 0] - self.footy_ref[k - 1, 0]
      self.COMy_es[k - 1, 0] = self.Lyy_ref[k - 1, 0] / 2
      self.COMvy_is[k - 1, 0] = (self.COMy_es[k - 1, 0] - self.COMy_is[k - 1, 0] * np.dot(S3, vari_ini)) / (
                  1 / Wn * np.dot(S4, vari_ini))

      for j in range(k + 1, self.Tn + 1):
          self.Tx[j - 1, 0] = self.Tx[j - 2, 0] + self.Ts[j - 2, 0]

      self.footx_ref[k, 0] = self.footx_ref[k - 1, 0] + np.dot(S1, vari_ini)
      self.footy_ref[k, 0] = self.footy_ref[k - 1, 0] + np.dot(S2, vari_ini)

      self.comvx_endref = Wn * self.COMx_is[k - 1, 0] * np.dot(S4, vari_ini) + self.COMvx_is[k - 1, 0] * np.dot(S3, vari_ini)
      self.comvy_endref = Wn * self.COMy_is[k - 1, 0] * np.dot(S4, vari_ini) + self.COMvy_is[k - 1, 0] * np.dot(S3, vari_ini)

      ## com trajectory generation of the next sampling time
      tj = self.dt
      cosh_wntj = np.cosh(Wn * tj)
      sinh_wntj = np.sinh(Wn * tj)
      self.comx[i, 0] = self.COMx_is[k - 1, 0] * cosh_wntj + self.COMvx_is[k - 1, 0] * 1 / Wn * sinh_wntj + self.px[i, 0]
      self.comy[i, 0] = self.COMy_is[k - 1, 0] * cosh_wntj + self.COMvy_is[k - 1, 0] * 1 / Wn * sinh_wntj + self.py[i, 0]
      self.comvx[i, 0] = Wn * self.COMx_is[k - 1, 0] * sinh_wntj + self.COMvx_is[k - 1, 0] * cosh_wntj
      self.comvy[i, 0] = Wn * self.COMy_is[k - 1, 0] * sinh_wntj + self.COMvy_is[k - 1, 0] * cosh_wntj
      self.comax[i, 0] = Wn ** 2 * self.COMx_is[k - 1, 0] * cosh_wntj + self.COMvx_is[k - 1, 0] * Wn * sinh_wntj
      self.comay[i, 0] = Wn ** 2 * self.COMy_is[k - 1, 0] * cosh_wntj + self.COMvy_is[k - 1, 0] * Wn * sinh_wntj
      self.cpx[i, 0] = self.comx[i, 0] + self.comvx[i, 0] / Wn
      self.cpy[i, 0] = self.comy[i, 0] + self.comvy[i, 0] / Wn
      self.cpx_T[i, 0] = self.COMx_es[k - 1, 0] + self.comvx_endref[0, 0] / Wn
      self.cpy_T[i, 0] = self.COMy_es[k - 1, 0] + self.comvy_endref[0, 0] / Wn

      ############################################################################
      ### state feedback ############
      # % % % external disturbances
      self.comx_feed[i, 0] = self.comx[i, 0]
      self.comvx_feed[i, 0] = self.comvx[i, 0]
      self.comax_feed[i, 0] = self.comax[i, 0]
      self.comy_feed[i, 0] = self.comy[i, 0]
      self.comvy_feed[i, 0] = self.comvy[i, 0]
      self.comay_feed[i, 0] = self.comay[i, 0]

      # if ((i==101) or (i==102) or (i==103)):
      #   if (i == 103):
      #     self.comx_feed[i,0] = comx[i,0] + det_xp / 4
      #     self.comvx_feed[i,0] = comvx[i,0] + det_xv / 2
      #     self.comy_feed[i,0] = comy[i,0] + det_yp / 4
      #     self.comvy_feed[i,0] = comvy[i,0] + det_yv / 2
      #   else:
      #     self.comx_feed[i,0] = comx[i,0] + det_xp / 2
      #     self.comvx_feed[i,0] = comvx[i,0] + det_xv * 1.5
      #     self.comy_feed[i,0] = comy[i,0] + det_yp / 2
      #     self.comvy_feed[i,0] = comvy[i,0] + det_yv * 1.5
      # else:
      #   if (i==40) or (i==41) or (i==42):
      #     if (i == 42):
      #       self.comx_feed[i,0] = comx[i,0] - det_xp / 4
      #       self.comvx_feed[i,0] = comvx[i,0] - det_xv / 2
      #       self.comy_feed[i,0] = comy[i,0] - det_yp / 4
      #       self.comvy_feed[i,0] = comvy[i,0] - det_yv / 2
      #     else:
      #       self.comx_feed[i,0] = comx[i,0] - det_xp / 2
      #       self.comvx_feed[i,0] = comvx[i,0] - det_xv / 2
      #       self.comy_feed[i,0] = comy[i,0] - det_yp / 2
      #       self.comvy_feed[i,0] = comvy[i,0] - det_yv / 2

      # time_end = time.time()
      # # print('self.footx_ref:\n', self.footx_ref)

      self.CoM_height(i)

      res_out = np.array(
          [self.Lxx_ref_real[i, 0], self.Lyy_ref_real[i, 0], self.Ts_ref_real[i, 0], self.comx[i, 0], self.comy[i, 0], self.comz[i, 0], self.px[i, 0],
           self.py[i, 0], self.cpx[i, 0], self.cpy[i, 0], self.cpx_T[i, 0], self.cpy_T[i, 0]])

      return res_out

  ###### CoM height polynomial interpolation##########################################
  def solve_AAA_inv_x(self, t_plan):
      AAA = np.zeros([7, 7])

      AAA[0, :] = np.array([6 * pow(t_plan[0], 5), 5 * pow(t_plan[0], 4), 4 * pow(t_plan[0], 3), 3 * pow(t_plan[0], 2),
                            2 * pow(t_plan[0], 1), 1, 0])  ### initial velocity

      AAA[1, :] = np.array(
          [30 * pow(t_plan[0], 4), 20 * pow(t_plan[0], 3), 12 * pow(t_plan[0], 2), 6 * pow(t_plan[0], 1), 2, 0,
           0])  ### initial acce

      AAA[2, :] = np.array(
          [pow(t_plan[0], 6), pow(t_plan[0], 5), pow(t_plan[0], 4), pow(t_plan[0], 3), pow(t_plan[0], 2),
           pow(t_plan[0], 1), 1])  ### initial pose

      AAA[3, :] = np.array(
          [pow(t_plan[1], 6), pow(t_plan[1], 5), pow(t_plan[1], 4), pow(t_plan[1], 3), pow(t_plan[1], 2),
           pow(t_plan[1], 1), 1])  ### middle position

      AAA[4, :] = np.array(
          [pow(t_plan[2], 6), pow(t_plan[2], 5), pow(t_plan[2], 4), pow(t_plan[2], 3), pow(t_plan[2], 2),
           pow(t_plan[2], 1), 1])  ### ending position

      AAA[5, :] = np.array(
          [30 * pow(t_plan[2], 4), 20 * pow(t_plan[2], 3), 12 * pow(t_plan[2], 2), 6 * pow(t_plan[2], 1), 2, 0,
           0])  ### ending acc

      AAA[6, :] = np.array([6 * pow(t_plan[2], 5), 5 * pow(t_plan[2], 4), 4 * pow(t_plan[2], 3), 3 * pow(t_plan[2], 2),
                            2 * pow(t_plan[2], 1), 1, 0])  ### ending vel

      AAA_inv = np.linalg.inv(AAA)

      return AAA_inv

  def CoM_height(self, i):

      bjx1 = self.index_find(i + 1, self.Tx, 0)

      ki = np.round(self.Tx[bjx1 - 1, 0] / self.dt)
      t_des = (i + 1 - ki) * self.dt  ####left time
      # t_remain = self.Ts[bjx1-1,0] - t_des

      t_plan = []
      #### planA: double support for height variation
      # if (bjx1>=2):
      #   if (t_des > td[bjx1-1,:]): ### SSP: no height variation
      #     comz[i, :] = self.footz_ref[bjx1-1,:] + hcom
      #     comz[i+1, :] = self.footz_ref[bjx1 - 1, :] + hcom
      #   else:
      #     t_plan = [0.0001, td[bjx1-1,:]/2+0.001, td[bjx1-1,:]+0.001]
      #
      #     aaa_inv = solve_AAA_inv_x(t_plan)
      #
      #     AAA_des = np.zeros([3, 7])
      #
      #     AAA_des[0, :] = np.array([6 * pow(t_des, 5), 5 * pow(t_des, 4), 4 * pow(t_des, 3), 3 * pow(t_des, 2), 2 * pow(t_des, 1), 1, 0])  ### desired velocity
      #
      #     AAA_des[1, :] = np.array([30 * pow(t_des, 4), 20 * pow(t_des, 3), 12 * pow(t_des, 2), 6 * pow(t_des, 1), 2, 0, 0])  ### desired acce
      #
      #     AAA_des[2, :] = np.array([pow(t_des, 6), pow(t_des, 5), pow(t_des, 4), pow(t_des, 3), pow(t_des, 2), pow(t_des, 1), 1])  ### desired pose
      #
      #     comz_plan = np.array([[0],[0],[self.footz_ref[bjx1-2,:] + hcom],[(self.footz_ref[bjx1-2,:] + self.footz_ref[bjx1-1,:])*0.5 + hcom],[self.footz_ref[bjx1-1,:] + hcom],[0],[0]])
      #
      #     comzvap = AAA_des.dot(aaa_inv.dot(comz_plan))
      #     comz[i, :] = comzvap[2, :]
      #     comvz[i, :] = comzvap[0, :]
      #     comaz[i, :] = comzvap[1, :]
      #
      #     comz[i+1, :] = comzvap[2, :] + comzvap[0, :] *dt + 0.5 * comzvap[0, :] * pow(dt,2)
      #
      # else:
      #   comz[i,:] = hcom
      #   comvz[i, :] = 0
      #   comaz[i, :] = 0

      #### planB: height variation duration whole period
      if (bjx1 >= 2):
          t_plan = [0.0001, self.Ts[bjx1 - 1, :] / 2 + 0.001, self.Ts[bjx1 - 1, :] + 0.001]

          aaa_inv = self.solve_AAA_inv_x(t_plan)

          AAA_des = np.zeros([3, 7])

          AAA_des[0, :] = np.array(
              [6 * pow(t_des, 5), 5 * pow(t_des, 4), 4 * pow(t_des, 3), 3 * pow(t_des, 2), 2 * pow(t_des, 1), 1,
               0])  ### desired velocity

          AAA_des[1, :] = np.array(
              [30 * pow(t_des, 4), 20 * pow(t_des, 3), 12 * pow(t_des, 2), 6 * pow(t_des, 1), 2, 0,
               0])  ### desired acce

          AAA_des[2, :] = np.array(
              [pow(t_des, 6), pow(t_des, 5), pow(t_des, 4), pow(t_des, 3), pow(t_des, 2), pow(t_des, 1),
               1])  ### desired pose

          comz_plan = np.array(
              [[0], [0], [self.footz_ref[bjx1 - 2, :] + self.hcom],
               [(self.footz_ref[bjx1 - 2, :] + self.footz_ref[bjx1 - 1, :]) * 0.5 + self.hcom],
               [self.footz_ref[bjx1 - 1, :] + self.hcom], [0], [0]])

          comzvap = AAA_des.dot(aaa_inv.dot(comz_plan))
          self.comz[i, :] = comzvap[2, :]
          self.comvz[i, :] = comzvap[0, :]
          self.comaz[i, :] = comzvap[1, :]

          self.comz[i + 1, :] = comzvap[2, :] + comzvap[0, :] * self.dt + 0.5 * comzvap[0, :] * pow(self.dt, 2)

      else:
          self.comz[i, :] = self.hcom
          self.comvz[i, :] = 0
          self.comaz[i, :] = 0

  ############ CoM-high frequency- generation #################################
  def XGetSolution_CoM_position(self, walktime, dt_sample, j_index):
      com_inte = np.zeros([9, 1])

      if (walktime >= 2):
          t_cur = walktime * dt_sample
          t_plan = [0, dt_sample, (j_index + 1) * self.dt - (t_cur - 2 * dt_sample)]

          AAA = np.zeros([4, 4])

          AAA[0, :] = np.array([pow(t_plan[0], 3), pow(t_plan[0], 2), pow(t_plan[0], 1), 1])  ### initial pose
          AAA[1, :] = np.array([pow(t_plan[1], 3), pow(t_plan[1], 2), pow(t_plan[1], 1), 1])  ### middle position
          AAA[2, :] = np.array([pow(t_plan[2], 3), pow(t_plan[2], 2), pow(t_plan[2], 1), 1])  ### ending position
          AAA[3, :] = np.array([3 * pow(t_plan[2], 2), 2 * pow(t_plan[2], 1), 1, 0])  ### ending vel

          AAA_inv = np.linalg.inv(AAA)

          t_a_plan = np.array([[pow(2 * dt_sample, 3), pow(2 * dt_sample, 2), pow(2 * dt_sample, 1), 1]])
          t_a_planv = np.array([[3*pow(2 * dt_sample, 2), 2*pow(2 * dt_sample, 1), 1, 0]])
          t_a_plana = np.array([[6*pow(2 * dt_sample, 1), 2, 0, 0]])

          tempx = np.array(
              [[self.comx1[walktime - 2, 0]], [self.comx1[walktime - 1, 0]], [self.comx[j_index, 0]], [self.comvx[j_index, 0]]])
          com_inte[0, :] = np.dot(t_a_plan, np.dot(AAA_inv, tempx))
          com_inte[3, :] = np.dot(t_a_planv, np.dot(AAA_inv, tempx))
          com_inte[6, :] = np.dot(t_a_plana, np.dot(AAA_inv, tempx))

          tempy = np.array(
              [[self.comy1[walktime - 2, 0]], [self.comy1[walktime - 1, 0]], [self.comy[j_index, 0]], [self.comvy[j_index, 0]]])
          com_inte[1, :] = np.dot(t_a_plan, np.dot(AAA_inv, tempy))
          com_inte[4, :] = np.dot(t_a_planv, np.dot(AAA_inv, tempy))
          com_inte[7, :] = np.dot(t_a_plana, np.dot(AAA_inv, tempy))

          tempz = np.array(
              [[self.comz1[walktime - 2, 0]], [self.comz1[walktime - 1, 0]], [self.comz[j_index, 0]], [self.comvz[j_index, 0]]])
          com_inte[2, :] = np.dot(t_a_plan, np.dot(AAA_inv, tempz))
          com_inte[5, :] = np.dot(t_a_planv, np.dot(AAA_inv, tempz))
          com_inte[8, :] = np.dot(t_a_plana, np.dot(AAA_inv, tempz))
      else:
          com_inte[0, 0] = copy.deepcopy(self.comx1[walktime - 1, 0])
          com_inte[1, 0] = copy.deepcopy(self.comy1[walktime - 1, 0])
          com_inte[2, 0] = copy.deepcopy(self.comz1[walktime - 1, 0])

      self.comx1[walktime, 0] = copy.deepcopy(com_inte[0, 0])
      self.comy1[walktime, 0] = copy.deepcopy(com_inte[1, 0])
      self.comz1[walktime, 0] = copy.deepcopy(com_inte[2, 0])

      com_inte[2, 0] -= self.hcom

      return com_inte

  def kmp_foot_trajectory(self, walktime, dt_sample, j_index, rleg_traj_refx, lleg_traj_refx, inDim, outDim, kh, lamda,
                          pvFlag):
      rffoot_traj = np.zeros([18, 1])
      right_leg_support = 0  ### 2:double support, 1:right_support, 0:left support

      rleg_traj_ref = copy.deepcopy(rleg_traj_refx)
      lleg_traj_ref = copy.deepcopy(lleg_traj_refx)
      row_num = rleg_traj_ref.shape[0]
      col_num = rleg_traj_ref.shape[1]
      ### dt_sample ==self.dtx,  j_index = i (nlp_gait),  walktime = (real_time loop number)
      ### lift height for foot trajectory

      lift_max = self.liftheight

      fooy_modeif = copy.deepcopy(self.footy_ref[0, 0])
      self.footy_ref[0, 0] = copy.deepcopy(self.footy_ref[2, 0])

      #### three via_points
      via_point1 = np.zeros([1, col_num])
      via_point1[0, 7] = 0.00000000001
      via_point1[0, 14] = 0.00000000001
      via_point1[0, 21] = 0.00000000001
      via_point1[0, 28] = 0.00000000001
      via_point1[0, 35] = 0.00000000001
      via_point1[0, 42] = 0.00000000001

      via_point2 = np.zeros([1, col_num])
      via_point2[0, 0] = 0.65 / 2
      via_point2[0, 7] = 0.00000000001
      via_point2[0, 14] = 0.00000000001
      via_point2[0, 21] = 0.00000000001
      via_point2[0, 28] = 0.00000000001
      via_point2[0, 35] = 0.00000000001
      via_point2[0, 42] = 0.00000000001

      via_point3 = np.zeros([1, col_num])
      via_point3[0, 0] = 0.65
      via_point3[0, 7] = 0.00000000001
      via_point3[0, 14] = 0.00000000001
      via_point3[0, 21] = 0.00000000001
      via_point3[0, 28] = 0.00000000001
      via_point3[0, 35] = 0.00000000001
      via_point3[0, 42] = 0.00000000001

      td = self.dsp * self.Ts
      bjx1 = self.index_find(j_index + 1, self.Tx, 0)
      t_des = (walktime + 3) * dt_sample - self.Tx[bjx1 - 1, 0] - td[bjx1 - 1, 0]  ####elapsed time

      # if (bjx1 > 12):
      #     lift_max +=0.01

      Rfpos = np.zeros([3, 1])
      Lfpos = np.zeros([3, 1])
      Rfposv = np.zeros([3, 1])
      Lfposv = np.zeros([3, 1])

      Rfpos[1, 0] = -abs(self.footy_offline[3, 0])
      Lfpos[1, 0] = abs(self.footy_offline[3, 0])

      if (bjx1 >= 2):
          if ((bjx1 % 2) == 0):  ##singular number: left support
              self.lfoot_kmp[0, 0] = self.footx_ref[bjx1 - 1, 0]
              self.lfoot_kmp[1, 0] = self.footy_ref[bjx1 - 1, 0]
              self.lfoot_kmp[2, 0] = self.footz_ref[bjx1 - 1, 0]
              self.lfootv_kmp[0, 0] = 0.0
              self.lfootv_kmp[1, 0] = 0.0
              self.lfootv_kmp[2, 0] = 0.0

              right_leg_support = 0

              if (t_des <= dt_sample):  ### double support phase
                  right_leg_support = 2
                  self.rfoot_kmp[0, 0] = self.footx_ref[bjx1 - 2, 0]
                  self.rfoot_kmp[1, 0] = self.footy_ref[bjx1 - 2, 0]
                  self.rfoot_kmp[2, 0] = self.footz_ref[bjx1 - 2, 0]
                  self.rfootv_kmp[0, 0] = 0.0
                  self.rfootv_kmp[1, 0] = 0.0
                  self.rfootv_kmp[2, 0] = 0.0

              else:  #####single support phase
                  t_des_k = 0.65 / (self.Ts[bjx1 - 1, 0] - td[bjx1 - 1, 0]) * t_des
                  if (t_des < 2 * dt_sample):
                      ### initialization
                      self.r_kmp = KMP(rleg_traj_ref, row_num, inDim, outDim, kh, lamda, pvFlag)

                      #### add via_point1#############
                      via_point1[0, 0] = 0.65 / (self.Ts[bjx1 - 1, 0] - td[bjx1 - 1, 0]) * (t_des - dt_sample)
                      via_point1[0, 1] = self.rfoot_kmp[0, 0] - self.footx_ref[bjx1 - 2, 0]
                      via_point1[0, 2] = self.rfoot_kmp[1, 0] - (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                      via_point1[0, 3] = self.rfoot_kmp[2, 0] - self.footz_ref[bjx1 - 2, 0]
                      via_point1[0, 4] = self.rfootv_kmp[0, 0]
                      via_point1[0, 5] = self.rfootv_kmp[1, 0]
                      via_point1[0, 6] = self.rfootv_kmp[2, 0]

                      self.r_kmp.kmp_point_insert(via_point1)

                      #### add via_point2#############
                      via_point2[0, 1] = (self.footx_ref[bjx1, 0] - self.footx_ref[bjx1 - 2, 0]) / 2
                      via_point2[0, 2] = (self.footy_ref[bjx1, 0] + self.footy_ref[bjx1 - 2, 0]) / 2 - (
                                  self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                      via_point2[0, 3] = (self.footz_ref[bjx1 - 1, 0] - self.footz_ref[bjx1 - 2, 0]) + lift_max
                      via_point2[0, 4] = (self.footx_ref[bjx1, 0] - self.footx_ref[bjx1 - 2, 0]) / 0.65 * 1.15
                      via_point2[0, 5] = (self.footy_ref[bjx1, 0] - self.footy_ref[bjx1 - 2, 0]) / 0.65
                      via_point2[0, 6] = 0
                      if (self.footz_ref[bjx1, 0] - self.footz_ref[bjx1 - 2, 0] <= 0):
                          via_point2[0, 6] = (self.footz_ref[bjx1, 0] - self.footz_ref[bjx1 - 2, 0]) / 0.65

                      self.r_kmp.kmp_point_insert(via_point2)

                      #### add via point3#########
                      via_point3[0, 1] = (self.footx_ref[bjx1, 0] - self.footx_ref[bjx1 - 2, 0])
                      via_point3[0, 2] = self.footy_ref[bjx1, 0] - (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                      via_point3[0, 3] = (self.footz_ref[bjx1, 0] - self.footz_ref[bjx1 - 2, 0])
                      via_point3[0, 4] = 0
                      via_point3[0, 5] = 0
                      via_point3[0, 6] = 0

                      self.r_kmp.kmp_point_insert(via_point3)

                      self.r_kmp.kmp_est_Matrix()
                      # print("via_point1:", via_point1[0, 0:7])
                      # print("via_point2:", via_point2[0, 0:7])
                      # print("via_point3:", via_point3[0, 0:7])
                  else:
                      if ((abs(self.Lxx_ref_real[j_index, 0] - self.Lxx_ref_real[j_index - 1, 0]) > 0.01) or (
                              abs(self.Lyy_ref_real[j_index, 0] - self.Lyy_ref_real[j_index - 1, 0]) > 0.01) or (
                              abs(self.Ts_ref_real[j_index, 0] - self.Ts_ref_real[j_index - 1, 0]) >= dt_sample)):
                          #### add via_point1#############
                          via_point1[0, 0] = 0.65 / (self.Ts[bjx1 - 1, 0] - td[bjx1 - 1, 0]) * t_des
                          via_point1[0, 1] = self.rfoot_kmp[0, 0] - self.footx_ref[bjx1 - 2, 0]
                          via_point1[0, 2] = self.rfoot_kmp[1, 0] - (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                          via_point1[0, 3] = self.rfoot_kmp[2, 0] - self.footz_ref[bjx1 - 2, 0]
                          via_point1[0, 4] = self.rfootv_kmp[0, 0]
                          via_point1[0, 5] = self.rfootv_kmp[1, 0]
                          via_point1[0, 6] = self.rfootv_kmp[2, 0]

                          self.r_kmp.kmp_point_insert(via_point1)
                          self.r_kmp.kmp_est_Matrix()

                  r_kmp_posvel = self.r_kmp.kmp_prediction(t_des_k)
                  self.rfoot_kmp[0, 0] = r_kmp_posvel[0] + self.footx_ref[bjx1 - 2, 0]
                  self.rfoot_kmp[1, 0] = r_kmp_posvel[1] + (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                  self.rfoot_kmp[2, 0] = r_kmp_posvel[2] + self.footz_ref[bjx1 - 2, 0]
                  self.rfootv_kmp[0, 0] = r_kmp_posvel[3]
                  self.rfootv_kmp[1, 0] = r_kmp_posvel[4]
                  self.rfootv_kmp[2, 0] = r_kmp_posvel[5]
          else:  ## right support
              self.rfoot_kmp[0, 0] = self.footx_ref[bjx1 - 1, 0]
              self.rfoot_kmp[1, 0] = self.footy_ref[bjx1 - 1, 0]
              self.rfoot_kmp[2, 0] = self.footz_ref[bjx1 - 1, 0]
              self.rfootv_kmp[0, 0] = 0.0
              self.rfootv_kmp[1, 0] = 0.0
              self.rfootv_kmp[2, 0] = 0.0

              right_leg_support = 1

              if (t_des <= dt_sample):  ### double support phase
                  self.lfoot_kmp[0, 0] = self.footx_ref[bjx1 - 2, 0]
                  self.lfoot_kmp[1, 0] = self.footy_ref[bjx1 - 2, 0]
                  self.lfoot_kmp[2, 0] = self.footz_ref[bjx1 - 2, 0]
                  self.lfootv_kmp[0, 0] = 0.0
                  self.lfootv_kmp[1, 0] = 0.0
                  self.lfootv_kmp[2, 0] = 0.0

                  right_leg_support = 2
              else:  #####single support phase
                  t_des_k = 0.65 / (self.Ts[bjx1 - 1, 0] - td[bjx1 - 1, 0]) * t_des
                  if (t_des < 2 * dt_sample):
                      ### initialization
                      self.l_kmp = KMP(lleg_traj_ref, row_num, inDim, outDim, kh, lamda, pvFlag)

                      #### add via_point1#############
                      via_point1[0, 0] = 0.65 / (self.Ts[bjx1 - 1, 0] - td[bjx1 - 1, 0]) * (t_des - dt_sample)
                      via_point1[0, 1] = self.lfoot_kmp[0, 0] - self.footx_ref[bjx1 - 2, 0]
                      via_point1[0, 2] = self.lfoot_kmp[1, 0] - (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                      via_point1[0, 3] = self.lfoot_kmp[2, 0] - self.footz_ref[bjx1 - 2, 0]
                      via_point1[0, 4] = self.lfootv_kmp[0, 0]
                      via_point1[0, 5] = self.lfootv_kmp[1, 0]
                      via_point1[0, 6] = self.lfootv_kmp[2, 0]

                      self.l_kmp.kmp_point_insert(via_point1)

                      #### add via_point2#############
                      via_point2[0, 1] = (self.footx_ref[bjx1, 0] - self.footx_ref[bjx1 - 2, 0]) / 2
                      via_point2[0, 2] = (self.footy_ref[bjx1, 0] + self.footy_ref[bjx1 - 2, 0]) / 2 - (
                                  self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                      via_point2[0, 3] = (self.footz_ref[bjx1 - 1, 0] - self.footz_ref[bjx1 - 2, 0]) + lift_max
                      via_point2[0, 4] = (self.footx_ref[bjx1, 0] - self.footx_ref[bjx1 - 2, 0]) / 0.65 * 1.15
                      via_point2[0, 5] = (self.footy_ref[bjx1, 0] - self.footy_ref[bjx1 - 2, 0]) / 0.65
                      via_point2[0, 6] = 0
                      if (self.footz_ref[bjx1, 0] - self.footz_ref[bjx1 - 2, 0] <= 0):
                          via_point2[0, 6] = (self.footz_ref[bjx1, 0] - self.footz_ref[bjx1 - 2, 0]) / 0.65

                      self.l_kmp.kmp_point_insert(via_point2)

                      #### add via point3#########
                      via_point3[0, 1] = (self.footx_ref[bjx1, 0] - self.footx_ref[bjx1 - 2, 0])
                      via_point3[0, 2] = self.footy_ref[bjx1, 0] - (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                      via_point3[0, 3] = (self.footz_ref[bjx1, 0] - self.footz_ref[bjx1 - 2, 0])
                      via_point3[0, 4] = 0
                      via_point3[0, 5] = 0
                      via_point3[0, 6] = 0

                      self.l_kmp.kmp_point_insert(via_point3)

                      # print("via_point1:", via_point1[0, 0:7])
                      # print("via_point2:", via_point2[0, 0:7])
                      # print("via_point3:", via_point3[0, 0:7])
                      # print("t_des_k:", t_des_k)

                      self.l_kmp.kmp_est_Matrix()
                  else:
                      if ((abs(self.Lxx_ref_real[j_index, 0] - self.Lxx_ref_real[j_index - 1, 0]) > 0.01) or (
                              abs(self.Lyy_ref_real[j_index, 0] - self.Lyy_ref_real[j_index - 1, 0]) > 0.01) or (
                              abs(self.Ts_ref_real[j_index, 0] - self.Ts_ref_real[j_index - 1, 0]) >= dt_sample)):
                          #### add via_point1#############
                          via_point1[0, 0] = 0.65 / (self.Ts[bjx1 - 1, 0] - td[bjx1 - 1, 0]) * t_des
                          via_point1[0, 1] = self.lfoot_kmp[0, 0] - self.footx_ref[bjx1 - 2, 0]
                          via_point1[0, 2] = self.lfoot_kmp[1, 0] - (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                          via_point1[0, 3] = self.lfoot_kmp[2, 0] - self.footz_ref[bjx1 - 2, 0]
                          via_point1[0, 4] = self.lfootv_kmp[0, 0]
                          via_point1[0, 5] = self.lfootv_kmp[1, 0]
                          via_point1[0, 6] = self.lfootv_kmp[2, 0]

                          self.l_kmp.kmp_point_insert(via_point1)
                          self.l_kmp.kmp_est_Matrix()

                  l_kmp_posvel = self.l_kmp.kmp_prediction(t_des_k)
                  self.lfoot_kmp[0, 0] = l_kmp_posvel[0] + self.footx_ref[bjx1 - 2, 0]
                  self.lfoot_kmp[1, 0] = l_kmp_posvel[1] + (self.footy_ref[bjx1 - 2, 0] - (-0.0726))
                  self.lfoot_kmp[2, 0] = l_kmp_posvel[2] + self.footz_ref[bjx1 - 2, 0]
                  self.lfootv_kmp[0, 0] = l_kmp_posvel[3]
                  self.lfootv_kmp[1, 0] = l_kmp_posvel[4]
                  self.lfootv_kmp[2, 0] = l_kmp_posvel[5]
                  # print("l_kmp_posvel:", l_kmp_posvel)
          Rfpos = copy.deepcopy(self.rfoot_kmp)
          Rfposv = copy.deepcopy(self.rfootv_kmp)

          Lfpos = copy.deepcopy(self.lfoot_kmp)
          Lfposv = copy.deepcopy(self.lfootv_kmp)
      else:
          right_leg_support = 2
          Rfpos[1, 0] = -abs(self.footy_offline[3, 0])
          Lfpos[1, 0] = abs(self.footy_offline[3, 0])

      # if (bjx1>=3):
      #   if (bjx1 % 2 ==0):
      #     # print("walktime:", walktime)
      #     # print("j_index:", j_index)
      #
      #     print("bjx1:", bjx1)
      #     print("t-des:",t_des)
      #     print("Rfpos",Rfpos[:,0].transpose())
      #     # print("Lfpos",Lfpos[:,0].transpose())
      #     print("xxxxxx:")
      #   else:
      #     # print("walktime:", walktime)
      #     # print("j_index:", j_index)
      #     # print("bjx1:", bjx1)
      #     print("t-des:",t_des)
      #     # print("Rfpos",Rfpos[:,0].transpose())
      #     print("Lfpos",Lfpos[:,0].transpose())
      #     # print("xxxxxx:")

      self.footy_ref[0, 0] = copy.deepcopy(fooy_modeif)


      rffoot_traj[0, 0] = Rfpos[0, 0]
      rffoot_traj[1, 0] = (Rfpos[1, 0] - (-abs(self.footy_offline[3, 0])))
      rffoot_traj[2, 0] = Rfpos[2, 0]

      rffoot_traj[3, 0] = Lfpos[0, 0]
      rffoot_traj[4, 0] = (Lfpos[1, 0] - (abs(self.footy_offline[3, 0])))
      rffoot_traj[5, 0] = Lfpos[2, 0]

      rffoot_traj[6, 0] = Rfposv[0, 0]
      rffoot_traj[7, 0] = Rfposv[1, 0]
      rffoot_traj[8, 0] = Rfposv[2, 0]

      rffoot_traj[9, 0] = Lfposv[0, 0]
      rffoot_traj[10, 0] = Lfposv[1, 0]
      rffoot_traj[11, 0] = Lfposv[2, 0]

      if (bjx1<=2):
          rffoot_traj[1, 0] = 0
          rffoot_traj[4, 0] = 0

      return rffoot_traj, right_leg_support
