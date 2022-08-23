#### python envir setup
from __future__ import print_function

import copy
import os
from os.path import dirname, join, abspath
import sys
import platform

from pathlib import Path

### pinocchio
import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from pino_robot_ik import CLIK                        #### IK solver
from robot_tracking_controller import Gait_Controller #### State estimate
from LIP_motion_planner import Gait                   #### Gait planner
from robot_dynamics import ForceCtrl                  #### Force controller
from robot_Grf import Force_dist                      #### Ground reaction force compute
from KMP_class import KMP
from NLP_sdp_class import NLP

##### numpy
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

##### pybullet
import pybullet
import pybullet_data
from sim_env import SimEnv
from sim_robot import SimRobot
import time
import math
import datetime

##### subprocess for external .exe
import subprocess

import scipy

################################ pinocchio urdf setup ##################################
def addFreeFlyerJointLimits(robot):
    rmodel = robot.model

    ub = rmodel.upperPositionLimit
    ub[:7] = 1e-6
    rmodel.upperPositionLimit = ub
    lb = rmodel.lowerPositionLimit
    lb[:7] = -1e-6
    rmodel.lowerPositionLimit = lb

############################ NMPC c++ run for gait generation ##########################################################
#convert string  to number
def str2num(LineString, comment='#'):
    from io import StringIO as StringIO
    import re, numpy

    NumArray = numpy.empty([0], numpy.int16)
    NumStr = LineString.strip()
    # ~ ignore comment string
    for cmt in comment:
        CmtRe = cmt + '.*$'
        NumStr = re.sub(CmtRe, " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    # ~ delete all non-number characters,replaced by blankspace.
    NumStr = re.sub('[^0-9.e+-]', " ", NumStr, count=0, flags=re.IGNORECASE)

    # ~ Remove incorrect combining-characters for double type.
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[.e+-](?=\s)', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)
    NumStr = re.sub('[e+-]$', " ", NumStr.strip(), count=0, flags=re.IGNORECASE)

    if len(NumStr.strip()) > 0:
        StrIOds = StringIO(NumStr.strip())
        NumArray = numpy.genfromtxt(StrIOds)

    return NumArray

def run_nmpc_external_ext(j,cpptest):
    b = str(j)
    # if os.path.exists(cpptest):
    rc, out = subprocess.getstatusoutput(cpptest + ' ' + b)
    donser = str2num(out)

    return donser

############## IK computing ####################################################################
def joint_lower_leg_ik(robot, oMdes_FL, JOINT_ID_FL, oMdes_FR, JOINT_ID_FR, oMdes_RL, JOINT_ID_RL, oMdes_RR, JOINT_ID_RR, Freebase, Homing_pose):
    ############ IK-solution ###############################################################33
    IK_FL_leg = CLIK(robot, oMdes_FL, JOINT_ID_FL, Freebase)
    IK_FR_leg = CLIK(robot, oMdes_FR, JOINT_ID_FR, Freebase)
    IK_RL_leg = CLIK(robot, oMdes_RL, JOINT_ID_RL, Freebase)
    IK_RR_leg = CLIK(robot, oMdes_RR, JOINT_ID_RR, Freebase)

    q = robot.q0
    if t < t_homing + 0.05:
        if Freebase:
            q[0 + 7:4 + 7] = Homing_pose[4:8]
            q[4 + 7:8 + 7] = Homing_pose[0:4]
            q[8 + 7:12 + 7] = Homing_pose[12:16]
            q[12 + 7:16 + 7] = Homing_pose[8:12]
        else:
            q = Homing_pose

    # ############### Jacobian-based IK: not used for quadrupedal#############################
    q_FL, J_FL = IK_FL_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # q_FR, J_FR = IK_FR_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # q_RL, J_RL = IK_RL_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    # q_RR, J_RR = IK_RR_leg.ik_Jacobian(q=q, Freebase=Freebase, eps=1e-6, IT_MAX=1000, DT=1e-1, damp=1e-6)
    ############################################################
    ############## nonlinear optimization based IK solvers#############################
    q_FL = IK_FL_leg.fbgs_opt(q)
    q_FR = IK_FR_leg.fbgs_opt(q)
    q_RL = IK_RL_leg.fbgs_opt(q)
    q_RR = IK_RR_leg.fbgs_opt(q)


    ##### transfer the pinocchio joint to pybullet joint##########################
    q_ik = Homing_pose
    if Freebase:
        q_ik[0:4] = q_FR[4 + 7:8 + 7]
        q_ik[4:8] = q_FL[0 + 7:4 + 7]
        q_ik[8:12] = q_RR[12 + 7:16 + 7]
        q_ik[12:16] = q_RL[8 + 7:12 + 7]

    return q_ik

############################################################################### robot setup ###############
########whold-body simu:
Full_body_simu = True
##########for robot with float-base ################################
Freebase = True

mesh_dir = str(Path(__file__).parent.absolute())
print("mesh_dir:",mesh_dir)

# You should change here to set up your own URDF file
if Full_body_simu:
    # urdf_filename = mesh_dir + '/go1_description/urdf/go1_full_no_grippers.urdf'
    urdf_filename = mesh_dir + '/go1_description/urdf/go1_origin.urdf'
else:
    urdf_filename = mesh_dir + '/go1_description/urdf/go1_origin.urdf'

### pinocchio load urdf
if Freebase:
    robot =  RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir, pin.JointModelFreeFlyer())
    addFreeFlyerJointLimits(robot)
else:
    robot = RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir)

####### only for debug check #########################################
# ## explore the model class
# for name, function in robot.model.__class__.__dict__.items():
#     print(' **** %s: %s' % (name, function.__doc__))
# print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444' )
# print('standard model: dim=' + str(len(robot.model.joints)))
# for jn in robot.model.joints:
#     print(jn)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' )

# find lower-leg joint idx in pinocchio;
id_FR =['FR_hip_joint','FR_thigh_joint','FR_calf_joint','FR_foot_fixed']
id_FL =['FL_hip_joint','FL_thigh_joint','FL_calf_joint','FL_foot_fixed']
id_RR =['RR_hip_joint','RR_thigh_joint','RR_calf_joint','RR_foot_fixed']
id_RL =['RL_hip_joint','RL_thigh_joint','RL_calf_joint','RL_foot_fixed']
idFR =[]
idFL =[]
idRR =[]
idRL =[]

for i in range(0,len(id_FR)):
    idFR.append(robot.model.getJointId(id_FR[i]))
    idFL.append(robot.model.getJointId(id_FL[i]))
    idRR.append(robot.model.getJointId(id_RR[i]))
    idRL.append(robot.model.getJointId(id_RL[i]))
print("FR leg joint id in pinocchio",idFR)
print("FL leg joint id in pinocchio",idFL)
print("RR leg joint id in pinocchio",idRR)
print("RL leg joint id in pinocchio",idRL)

##### reset base pos and orn,  only workable in floating-based model
# robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array([1,0.5,0.8]))
q = robot.q0
print("q:", q)

############################### pinocchio load finish !!!!!!!!!!!!!!!!!!!!!!!!!!! #####################################################
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pinocchio setup finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")

###################################################
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!set robot controller !!!!!!!!!!!!!!!!!!!!!!!!!!")
################################################### pybullet simulation setup ###########################
### intial pose for go1 in pybullet
sim_rate = 250
print("control frequency:",sim_rate)
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate, real_time_sim =False)
sim_env.resetCamera()

trailDuration = 10
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0.446]
hasPrevPose = 0
urobtx = SimRobot(urdfFileName=urdf_filename,
                 basePosition=prevPose1,
                 baseRPY=[0, 0, 0],
                 Torquecontrol = False)
go1id = urobtx.id

num_joints = urobtx.getNumJoints()
num_actuated_joint = urobtx.getNumActuatedJoints()
joint_perLeg = int(num_actuated_joint/4.0)
actuation_joint_index = urobtx.getActuatedJointIndexes()

### Homing_pose: four legs:FR, FL, RR, RL####### is important for walking in place
Homing_pose = np.zeros(num_actuated_joint)
for jy in range(0,4):
    Homing_pose[jy * joint_perLeg + 0] = 0
    Homing_pose[jy * joint_perLeg + 1] = 0.8
    Homing_pose[jy * joint_perLeg + 2] = -1.3
print("Homing_pose:",Homing_pose)

Homing_height_reduce = 0.1  ####bending knee

q_cmd = np.zeros(num_actuated_joint)
q_cmd_pre = np.zeros(num_actuated_joint)
q_vel_cmd = np.zeros(num_actuated_joint)
torque_cmd = np.zeros(num_actuated_joint)
t_homing = 1
n_t_homing = round(t_homing/dt)

useRealTimeSimulation = 0
pybullet.setRealTimeSimulation(useRealTimeSimulation)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pybullet load environment finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")


######################################################## Gait mode ###########################
Gait_mode = 1 ########## 0:pace(bipedal); 1: troting; 2:bounding
if(Gait_mode==0):
    y_lamda = 0.595
else:
    if(Gait_mode==1):
        y_lamda = 0
    else:
        y_lamda = 1.15
######################################################## Gait mode ###########################



######################################################
##### Gait_Controller_estimation #####################################################
State_estimator = Gait_Controller(urbodx = urobtx, id = go1id,gait_mode=Gait_mode,verbose=True)
##### Force controller
Force_controller = ForceCtrl(urbodx = urobtx)

t=0.
leg_FL_homing_fix = []
leg_FR_homing_fix = []
leg_RL_homing_fix = []
leg_RR_homing_fix = []
base_home_fix = []

torso1_linkid = 0
left_sole_linkid = 36
right_sole_linkid = 42
##############For kinematics
### desired leg position and velocity ##############################
des_FR_p = np.zeros([3,1])
des_FL_p = np.zeros([3,1])
des_RR_p = np.zeros([3,1])
des_RL_p = np.zeros([3,1])
des_FR_v = np.zeros([3,1])
des_FL_v = np.zeros([3,1])
des_RR_v = np.zeros([3,1])
des_RL_v = np.zeros([3,1])

### initialization for sim_robots, only once for FK
des_FL = np.array([3,1,1])
oMdes_FL = pin.SE3(np.eye(3), des_FL)
JOINT_ID_FL = idFL[-1]
#########leg IK ###############3
IK_leg = CLIK(robot, oMdes_FL, JOINT_ID_FL, Freebase)

##########gait planner initialization#################################################
#####################################################################################################33
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Walking parameter initialization !!!!!!!!!!!!!!!!!!!!!!!!!!")
#####################################################################################################33

mesh_dirx = str(Path(__file__).parent.absolute())
# angle_cmd_filename = mesh_dirx + '/go1/go1_angle_nmpc.txt'
# torque_cmd_filename = mesh_dirx + '/go1/go1_ref_torque_nmpc.txt'
#
# joint_angle_cmd = np.loadtxt(angle_cmd_filename)  ### linux/ubuntu
# joint_torque_cmd = np.loadtxt(torque_cmd_filename)  ### linux/ubuntu
#
# print(type(joint_angle_cmd))
# row_num = joint_angle_cmd.shape[0]
# col_num = joint_angle_cmd.shape[1]
# print(col_num)

demo_filename = mesh_dirx + '/referdata_swing_mod.txt'

nstep = 20
dt_sample = dt
dt_nlp = 0.05
Gait_home_height = 0.28
sx_x = 0.06
sy_y = 0.12675*2
sz_z = 0
st_t = 0.499
falling_flag = 0


if(Gait_mode==0):
    Dsp_rate = 0.15
    lift_height =0.015
elif(Gait_mode==1):
    Dsp_rate = 0.15
    lift_height =0.015
else:
    Dsp_rate = 0.1
    lift_height = 0.03
    sx_x = 0


Gait_func = Gait(st_t, sx_x, sy_y, sz_z, lift_height, nstep, Dsp_rate, dt, Qudrupedal=True)
N_gait = Gait_func.step_location()
N_gait += n_t_homing


pos_base = np.zeros([11,1])
Rlfoot_pos = np.zeros([18, 1])
right_leg_support = 2  ### 2:double support, 1:right_support, 0:left support

#####################################################################################################33
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!LIP-CLASS setup finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")
#####################################################################################################33


### KMP initiallization
rleg_traj_refx = np.loadtxt(demo_filename)  ### linux/ubuntu
lleg_traj_refx = np.loadtxt(demo_filename)  ### linux/ubuntu

inDim = 1  ### time as input
outDim = 6  ### decided by traj_Dim * (pos+?vel+?acc: indicated by pvFlag)
kh = 2
lamda = 1
pvFlag = 1  ## pvFlag = 0(pos) & pvFlag = 1 (pos+vel)

com_fo_nlp = NLP(nstep,dt_sample,dt_nlp,Gait_home_height,Dsp_rate,sx_x,sy_y,sz_z,st_t,lift_height,
                 rleg_traj_refx, lleg_traj_refx, inDim, outDim, kh, lamda, pvFlag,Gait_mode)
print("sampling_time_total:",com_fo_nlp.Nsum1)

t_n = int(2 * np.round(com_fo_nlp.Ts[0, 0] / dt_sample))


outx = np.zeros([N_gait, 15])
res_outx = np.zeros([1, 12])
com_intex = np.zeros([9,1])
RLfoot_com_pos = np.zeros([18, N_gait])
#####################################################################################################33
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NLP-KMP-CLASS setup finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")
#####################################################################################################33


##########
Grf_computer = Force_dist(urobtx)

full_joint_number = urobtx.getNumJoints()
FileLength = N_gait
traj_opt = np.zeros([FileLength,12])  ### trajectory reference generated by gait planner
joint_opt = np.zeros([FileLength,num_actuated_joint])  #### joint angle by IK
state_feedback = np.zeros([FileLength,40])   ### robot state estimation
links_pos_prev = np.zeros([full_joint_number+1,3])  ### links com position; the last one is the base position
links_vel_prev = np.zeros([full_joint_number+1,3])  ### links com velocities
support_flag = np.zeros([FileLength,1])
gcom_pre = [0,0,0]
com_ref_base  = [0,0,0]
com_feedback_base  = [0,0,0]
com_ref_det = [0,0,0]
com_ref_det_pre = [0,0,0]
com_feedback_det= [0,0,0]
com_feedback_det_pre= [0,0,0]

angle_ref_det= [0,0,0]
angle_feedback_det= [0,0,0]
angle_ref_det_pre= [0,0,0]
angle_feedback_det_pre= [0,0,0]

body_p_ref = np.zeros([3, 1])
body_v_ref = np.zeros([3, 1])
body_p_mea = np.zeros([3, 1])
body_v_mea = np.zeros([3, 1])
body_r_ref = np.zeros([3, 1])
body_rv_ref = np.zeros([3, 1])
body_r_mea = np.zeros([3, 1])
body_rv_mea = np.zeros([3, 1])
body_r_mea_pre = np.zeros([3, 1])

# desired foot force
Force_FR = np.zeros([3,1])
Force_FL = np.zeros([3,1])
Force_RR = np.zeros([3,1])
Force_RL = np.zeros([3,1])
Force_FR_old = np.zeros([3,1])
Force_FL_old = np.zeros([3,1])
Force_RR_old = np.zeros([3,1])
Force_RL_old = np.zeros([3,1])

q_FR = np.zeros(3)
q_FL = np.zeros(3)
q_RR = np.zeros(3)
q_RL = np.zeros(3)
Q_cmd = np.zeros([num_actuated_joint,N_gait])
Q_measure = np.zeros([num_actuated_joint,N_gait])
Q_velocity_measure = np.zeros([num_actuated_joint,N_gait])

leg_FR_homing = np.zeros([3,1])
leg_FL_homing = np.zeros([3,1])
leg_RR_homing = np.zeros([3,1])
leg_RL_homing = np.zeros([3,1])
leg_FR_homingx = np.zeros([3,1])
leg_FL_homingx = np.zeros([3,1])
leg_RR_homingx = np.zeros([3,1])
leg_RL_homingx = np.zeros([3,1])

J_FR = np.zeros([3,3])
J_FL = np.zeros([3,3])
J_RR = np.zeros([3,3])
J_RL = np.zeros([3,3])
des_base = np.array([0, 0, 0.446])
des_base_fix = np.array([0, 0, 0.446])
des_base_vel = np.zeros(3)
base_R = np.zeros([3,1])
h_com = 0
Torque_cmd = np.zeros([num_actuated_joint,N_gait])
Torque_FB = np.zeros([num_actuated_joint,N_gait])


det_torque_pd = np.zeros([num_actuated_joint,1])
############## Sample Gains
# joint PD gains
kp = np.array([100, 50, 50])
kd = np.array([1, 0.1, 0.1])*0.1


Torque_measured = np.zeros([num_actuated_joint,N_gait])
FSR_measured = np.zeros([num_actuated_joint,N_gait])
Pos_ref = np.zeros([18,N_gait])
Grf_ref = np.zeros([18,N_gait])
Grf_opt = np.zeros([12,N_gait])

Torque_gravity = 0.65
Gravity_comp = np.zeros(num_actuated_joint)
Gravity_comp[0] = -Torque_gravity
Gravity_comp[joint_perLeg] = Torque_gravity
Gravity_comp[2*joint_perLeg] = -Torque_gravity
Gravity_comp[3*joint_perLeg] = Torque_gravity
#### relative to base framework
Relative_FR_pos = np.zeros(3)
Relative_FR_vel = np.zeros(3)
Relative_FL_pos = np.zeros(3)
Relative_FL_vel = np.zeros(3)
Relative_RR_pos = np.zeros(3)
Relative_RR_vel = np.zeros(3)
Relative_RL_pos = np.zeros(3)
Relative_RL_vel = np.zeros(3)
Relative_FR_pos_mea = np.zeros(3)
Relative_FR_vel_mea = np.zeros(3)
Relative_FL_pos_mea = np.zeros(3)
Relative_FL_vel_mea = np.zeros(3)
Relative_RR_pos_mea = np.zeros(3)
Relative_RR_vel_mea = np.zeros(3)
Relative_RL_pos_mea = np.zeros(3)
Relative_RL_vel_mea = np.zeros(3)
### foot support stance
FR_support = True
FL_support = True
RR_support = True
RL_support = True

#####

### for fsr tracking controller
Force_tracking_kd = 0.001 * np.ones(3)
Force_tracking_kd[2] = 0.001

#### impedance control
IP_Force_FR = np.zeros([3,1])
IP_Force_FL = np.zeros([3,1])
IP_Force_RR = np.zeros([3,1])
IP_Force_RL = np.zeros([3,1])
actuatedJointtorques = np.zeros(num_actuated_joint)
####################################### main loop for robot gait generation and control ####################
i = 1
tx = 0

i_total = N_gait
#i_total = 2*n_t_homing + 200 + 2*(int)(round(st_t/dt))
while i<=i_total:



    if (useRealTimeSimulation):
        t = t + dt
    else:
        t = t + dt
    #################################### state esimation ##############################################################
    gcom_mea, FR_sole_pose_mea, FL_sole_pose_mea, RR_sole_pose_mea, RL_sole_pose_mea, base_pos_mea, base_angle_mea = State_estimator.cal_com_state()
    links_pos, links_vel, links_acc = State_estimator.get_link_vel_vol(i,dt,links_pos_prev,links_vel_prev)



    State_estimator.ankle_joint_pressure()
    q_mea = urobtx.getActuatedJointPositions()
    dq_mea = urobtx.getActuatedJointVelocities()
    T_mea = urobtx.getActuatedJointtorques()

    bas_pos,bas_ori = pybullet.getBasePositionAndOrientation(go1id)
    # print("bas_ori:",bas_ori)
    bas_posv, bas_oriv = pybullet.getBaseVelocity(go1id)
    measure_base = np.array(bas_pos)
    bas_eular = pybullet.getEulerFromQuaternion(bas_ori)
    for jx in range(0,3):
        base_R[jx,0] = base_angle_mea[jx]
    q_FR = q_mea[0:3]
    q_FL = q_mea[3:6]
    q_RR = q_mea[6:9]
    q_RL = q_mea[9:12]
    leg_FR, J_FR = IK_leg.fk_close_form(measure_base, base_R, q_FR, 0)
    leg_FL, J_FL = IK_leg.fk_close_form(measure_base, base_R, q_FL, 1)
    leg_RR, J_RR = IK_leg.fk_close_form(measure_base, base_R, q_RR, 2)
    leg_RL, J_RL = IK_leg.fk_close_form(measure_base, base_R, q_RL, 3)

    for j in range(0, 3):
        Relative_FR_pos_mea[j] = links_pos[State_estimator.FR_soleid, j] - bas_pos[j]
        Relative_FR_vel_mea[j] = links_vel[State_estimator.FR_soleid, j] - bas_posv[j]
        Relative_FL_pos_mea[j] = links_pos[State_estimator.FL_soleid, j] - bas_pos[j]
        Relative_FL_vel_mea[j] = links_vel[State_estimator.FL_soleid, j] - bas_posv[j]
        Relative_RR_pos_mea[j] = links_pos[State_estimator.RR_soleid, j] - bas_pos[j]
        Relative_RR_vel_mea[j] = links_vel[State_estimator.RR_soleid, j] - bas_posv[j]
        Relative_RL_pos_mea[j] = links_pos[State_estimator.RL_soleid, j] - bas_pos[j]
        Relative_RL_vel_mea[j] = links_vel[State_estimator.RL_soleid, j] - bas_posv[j]

    L_leg_p_mea = np.zeros([3,1])
    R_leg_p_mea = np.zeros([3,1])
    for j in range(0,3):
        R_leg_p_mea[j,0] = (links_pos[State_estimator.FR_soleid, j] + links_pos[State_estimator.RR_soleid, j])/2
        L_leg_p_mea[j,0] = (links_pos[State_estimator.FL_soleid, j] + links_pos[State_estimator.RL_soleid, j])/2


    state_feedback[i-1, 0:3] = gcom_mea
    state_feedback[i-1, 3:6] = (R_leg_p_mea[0:3,0]).T
    state_feedback[i-1, 6:9] = (L_leg_p_mea[0:3,0]).T
    # state_feedback[i, 9:15] = right_ankle_force
    # state_feedback[i, 15:21] = left_ankle_force
    # state_feedback[i, 21:24] = gcop_m
    # state_feedback[i, 24:27] = dcm_pos_m
    # state_feedback[i, 27:30] = com_vel_m
    state_feedback[i-1, 30:33] = base_pos_mea
    state_feedback[i-1, 33:36] = base_angle_mea

    FR_fsr = np.array(State_estimator.FR_sensor[0:3])
    FL_fsr = np.array(State_estimator.FL_sensor[0:3])
    RR_fsr = np.array(State_estimator.RR_sensor[0:3])
    RL_fsr = np.array(State_estimator.RL_sensor[0:3])
    FSR_measured[0:3,i-1] = FR_fsr
    FSR_measured[3:6,i-1] = FL_fsr
    FSR_measured[6:9,i-1] = RR_fsr
    FSR_measured[9:12,i-1] = RL_fsr

    Q_measure[:, i - 1] = q_mea
    Q_velocity_measure[:, i - 1] = q_mea
    Torque_measured[:, i - 1] = T_mea

    ## stage 1:  homing pose initialization ###########
    if i<=n_t_homing:
        #### initial state for homing_pose
        Homing_pose_t = Homing_pose*math.sin(t/t_homing/2.*math.pi)
        ####### forward state calculation #############
        base_pos_m = prevPose1
        body_R = np.zeros([3,1])
        if (i<=1): ####calculate the inital position of four legs
            q_FR = Homing_pose_t[0:3]
            q_FL = Homing_pose_t[3:6]
            q_RR = Homing_pose_t[6:9]
            q_RL = Homing_pose_t[9:12]
            leg_FR_homing, J_FR = IK_leg.fk_close_form(base_pos_m, body_R, q_FR, 0)
            leg_FL_homing, J_FL = IK_leg.fk_close_form(base_pos_m, body_R, q_FL, 1)
            leg_RR_homing, J_RR = IK_leg.fk_close_form(base_pos_m, body_R, q_RR, 2)
            leg_RL_homing, J_RL = IK_leg.fk_close_form(base_pos_m, body_R, q_RL, 3)
            h_com = base_pos_m[2]-(leg_FR_homing[2,0] + leg_FL_homing[2,0] + leg_RR_homing[2,0] + leg_RL_homing[2,0])/4


        des_FR_p = copy.deepcopy(leg_FR_homing)
        des_FL_p = copy.deepcopy(leg_FL_homing)
        des_RR_p = copy.deepcopy(leg_RR_homing)
        des_RL_p = copy.deepcopy(leg_RL_homing)

        ##### CoM trajectory: height variation
        Homing_height_t = h_com - Homing_height_reduce * math.sin(t / t_homing / 2. * math.pi)
        Homing_height_velt = -(1/t_homing/2 * math.pi) * Homing_height_reduce * math.cos(t / t_homing / 2. * math.pi)
        Homing_height_acct = (1/t_homing/2*math.pi)**2 * Homing_height_reduce * math.sin(t / t_homing / 2. * math.pi)

        des_base[2] = (Homing_height_t +(leg_FR_homing[2,0]+leg_FL_homing[2,0]+leg_RR_homing[2,0]+leg_RL_homing[2,0])/4)
        des_base_vel[2] = Homing_height_velt
        des_base_fix = copy.deepcopy(des_base)
        ############ Ik################
        body_R_des = np.zeros([3, 1])
        q_FR, J_FR = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max=15, lamda=0.55)
        q_FL, J_FL = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
        q_RR, J_RR = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max=15, lamda=0.55)
        q_RL, J_RL = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)
        Q_cmd[0:3,i-1] = q_FR[0:3]
        Q_cmd[3:6,i-1] = q_FL[0:3]
        Q_cmd[6:9,i-1] = q_RR[0:3]
        Q_cmd[9:12,i-1] = q_RL[0:3]

        ##################################### feedback control preparation#######################################
        for j in range(0,3):
            Relative_FR_pos[j] = des_FR_p[j,0] - des_base[j]
            Relative_FR_vel[j] = des_FR_v[j,0] - des_base_vel[j]
            Relative_FL_pos[j] = des_FL_p[j,0] - des_base[j]
            Relative_FL_vel[j] = des_FL_v[j,0] - des_base_vel[j]
            Relative_RR_pos[j] = des_RR_p[j,0] - des_base[j]
            Relative_RR_vel[j] = des_RR_v[j,0] - des_base_vel[j]
            Relative_RL_pos[j] = des_RL_p[j,0] - des_base[j]
            Relative_RL_vel[j] = des_RL_v[j,0] - des_base_vel[j]

        # control_mode: should be  positionControl mode
        if(urobtx.controlMode == 'positionControl'):
            if (i <= 10):  ## add external torques to maintain balance since starte from singular configuration
                pybullet.applyExternalTorque(go1id, 0, torqueObj=[0, -10, 0], flags=1)

            urobtx.setActuatedJointPositions(Q_cmd[:,i-1])
            q_cmd = Q_cmd[:,i-1]
        else:
            Torque_cmd[:, i - 1] = torque_cmd
            urobtx.setActuatedJointTorques(torque_cmd)

        support_flag[i] = 0  ### 0, double support, right support
    else:
        ################################## force controller test ###################################################
        ij = i - n_t_homing
        torque_cmd = urobtx.getActuatedJointtorques()
        if(ij<50):
            urobtx.setActuatedJointPositions(q_cmd)
            Q_cmd[:, i - 1] = q_cmd
            Torque_cmd[:, i - 1] = torque_cmd
        else:
            if(ij<200):#####falling down
                torque_cmd = np.zeros(num_actuated_joint)
                urobtx.setActuatedJointTorques(torque_cmd)

                des_basex = pybullet.getBasePositionAndOrientation(go1id)[0]
                des_base = np.array(des_basex)

                des_FR_p[:, 0] = (links_pos[State_estimator.FR_soleid, :]).T
                des_FL_p[:, 0] = (links_pos[State_estimator.FL_soleid, :]).T
                des_RR_p[:, 0] = (links_pos[State_estimator.RR_soleid, :]).T
                des_RL_p[:, 0] = (links_pos[State_estimator.RL_soleid, :]).T

                Q_cmd[:, i - 1] = q_cmd
                Torque_cmd[:, i - 1] = torque_cmd
            else: ######switching to torque control mode
                tx = t - (t_homing + 200 * dt)

                ########gait planning #######################
                if(ij==200):
                    des_basex = pybullet.getBasePositionAndOrientation(go1id)[0]

                    des_base = np.array(des_basex)
                    des_FR_p[:, 0] = (links_pos[State_estimator.FR_soleid, :]).T
                    des_FL_p[:, 0] = (links_pos[State_estimator.FL_soleid, :]).T
                    des_RR_p[:, 0] = (links_pos[State_estimator.RR_soleid, :]).T
                    des_RL_p[:, 0] = (links_pos[State_estimator.RL_soleid, :]).T

                    leg_FR_homingx[:, 0] = (links_pos[State_estimator.FR_soleid, :]).T
                    leg_FL_homingx[:, 0] = (links_pos[State_estimator.FL_soleid, :]).T
                    leg_RR_homingx[:, 0] = (links_pos[State_estimator.RR_soleid, :]).T
                    leg_RL_homingx[:, 0] = (links_pos[State_estimator.RL_soleid, :]).T

                    # str = input("Would go the dynamic mode!!!Press enter for start:")
                    # time.sleep(3)

                if(tx>t_homing):

                    #### gait planning-based on LIP=== testing troting gait #################################################
                    j_index = int(i - n_t_homing - 200)

                    # pos_base = Gait_func.Ref_com_lip_update(j_index,Homing_height_t)
                    # Rlfoot_pos, right_leg_support = Gait_func.FootpR(j_index)
                    # print("Rlfoot_pos:",Rlfoot_pos.T)

                    ################################################### NLP+KMP gait generation: using for locomotion control #################
                    j_ind = int(np.floor((j_index) / (com_fo_nlp.dt / com_fo_nlp.dtx)))  ####walking time fall into a specific optmization loop

                    if ((j_ind >= 1) and (abs(j_index * com_fo_nlp.dtx - j_ind * com_fo_nlp.dt) <= 0.8 * dt_sample)):
                        res_outx = com_fo_nlp.nlp_gait(j_ind,Gait_mode)

                    Rlfoot_pos, right_leg_support = com_fo_nlp.kmp_foot_trajectory(j_index, dt_sample, j_ind, rleg_traj_refx,lleg_traj_refx,
                                                                                   inDim, outDim, kh, lamda, pvFlag)
                    com_intex = com_fo_nlp.XGetSolution_CoM_position(j_index, dt_sample, j_ind)
                    pos_base[0:9,:] =  com_intex[0:9,:]

                    ##### troting test: no lateral movement ###
                    des_base[0] = des_base_fix[0] + pos_base[0, 0]
                    if(Gait_mode==1): ###troting
                        des_base[1] = des_base_fix[1] + pos_base[1, 0] * y_lamda
                    elif(Gait_mode==0): ###pacing
                        des_base[1] = des_base_fix[1] + pos_base[1, 0] * y_lamda
                    else:               ###bounding
                        des_base[0] = des_base_fix[0] + pos_base[1, 0] * y_lamda
                        des_base[1] = des_base_fix[1] + 0

                    des_base[2] = Gait_home_height + pos_base[2, 0]
                    des_base_vel = pos_base[3:6,0]
                    if (Gait_mode == 1):  ###troting
                        des_base_vel[1] = pos_base[4,0] * y_lamda
                    elif (Gait_mode == 0):  ###pacing
                        des_base_vel[1] = pos_base[4, 0] * y_lamda
                    else:               ###bounding
                        des_base_vel[0] = pos_base[4, 0] * y_lamda
                        des_base_vel[1] = 0

                    base_acc = pos_base[6:9,:]
                    if (Gait_mode == 1):  ###troting
                        base_acc[1] = pos_base[7,:] * y_lamda
                    elif (Gait_mode == 0):  ###pacing
                        base_acc[1] = pos_base[7,:] * y_lamda
                    else:               ###bounding
                        base_acc[0] = pos_base[7, :] * y_lamda
                        base_acc[1] = 0

                    base_p = np.zeros([3,1])
                    base_p[:,0] = des_base.T
                    cop_xyz = np.zeros([3,1])
                    cop_xyz[0:2,:] = pos_base[9:11,:]

                    # Rlfoot_pos[1, 0] = 0
                    # Rlfoot_pos[4, 0] = 0
                    if(Gait_mode==1): #### troting :FR--RL-left pair; :FL--RR-right pair
                        des_FR_p[0:3,0] = leg_FR_homing[0:3,0] + Rlfoot_pos[3:6, 0]
                        des_RL_p[0:3,0] = leg_RL_homing[0:3,0] + Rlfoot_pos[3:6, 0]
                        des_FL_p[0:3, 0] = leg_FL_homing[0:3,0] + Rlfoot_pos[0:3, 0]
                        des_RR_p[0:3, 0] = leg_RR_homing[0:3,0] + Rlfoot_pos[0:3, 0]
                        des_FR_v[0:3, 0] = Rlfoot_pos[9:12, 0]
                        des_RL_v[0:3, 0] = Rlfoot_pos[9:12, 0]
                        des_FL_v[0:3, 0] = Rlfoot_pos[6:9, 0]
                        des_RR_v[0:3, 0] = Rlfoot_pos[6:9, 0]

                        if(right_leg_support==1): ###right leg pair support
                            FR_support = False
                            RL_support = False
                            FL_support = True
                            RR_support = True
                        elif(right_leg_support==0): #####left leg pair support
                            FR_support = True
                            RL_support = True
                            FL_support = False
                            RR_support = False
                        else:
                            FR_support = True
                            RL_support = True
                            FL_support = True
                            RR_support = True
                    elif(Gait_mode==0): #### pacing :FL--RL-left pair; :FR--RR-right pair
                        # print("homing_foop,y",leg_FR_homing[1,0],leg_FL_homing[1,0],leg_RR_homing[1,0],leg_RL_homing[1,0])
                        des_FL_p[0:3,0] = leg_FL_homing[0:3,0] + Rlfoot_pos[3:6, 0]
                        des_RL_p[0:3,0] = leg_RL_homing[0:3,0] + Rlfoot_pos[3:6, 0]
                        des_FR_p[0:3, 0] = leg_FR_homing[0:3,0] + Rlfoot_pos[0:3, 0]
                        des_RR_p[0:3, 0] = leg_RR_homing[0:3,0] + Rlfoot_pos[0:3, 0]
                        des_FL_v[0:3, 0] = Rlfoot_pos[9:12, 0]
                        des_RL_v[0:3, 0] = Rlfoot_pos[9:12, 0]
                        des_FR_v[0:3, 0] = Rlfoot_pos[6:9, 0]
                        des_RR_v[0:3, 0] = Rlfoot_pos[6:9, 0]

                        if(right_leg_support==1): ###right leg pair support
                            FR_support = True
                            RR_support = True
                            FL_support = False
                            RL_support = False
                        elif(right_leg_support==0): #####left leg pair support
                            FR_support = False
                            RR_support = False
                            FL_support = True
                            RL_support = True
                        else:
                            FR_support = True
                            RL_support = True
                            FL_support = True
                            RR_support = True
                    else:  #### bounding :FL--FR-left pair; RL--RR-right pair
                        des_RL_p[0:3, 0] = leg_RL_homing[0:3, 0] + Rlfoot_pos[0:3, 0]
                        des_RR_p[0:3, 0] = leg_RR_homing[0:3, 0] + Rlfoot_pos[0:3, 0]
                        des_FL_p[0:3, 0] = leg_FL_homing[0:3, 0] + Rlfoot_pos[3:6, 0]
                        des_FR_p[0:3, 0] = leg_FR_homing[0:3, 0] + Rlfoot_pos[3:6, 0]
                        des_RL_v[0:3, 0] = Rlfoot_pos[6:9, 0]
                        des_RR_v[0:3, 0] = Rlfoot_pos[6:9, 0]
                        des_FL_v[0:3, 0] = Rlfoot_pos[9:12, 0]
                        des_FR_v[0:3, 0] = Rlfoot_pos[9:12, 0]
                        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                        if (right_leg_support == 1):  ###right leg pair support
                            FR_support = False
                            FL_support = False
                            RR_support = True
                            RL_support = True
                        elif (right_leg_support == 0):  #####left leg pair support
                            FR_support = True
                            FL_support = True
                            RR_support = False
                            RL_support = False
                        else:
                            FR_support = True
                            FL_support = True
                            RR_support = True
                            RL_support = True

                    body_R_des = np.zeros([3, 1])
                    body_Rv_des = np.zeros([3, 1])
                    base_R_acc = np.zeros([3, 1])

                    R_leg_p = np.zeros([3, 1])
                    L_leg_p = np.zeros([3, 1])
                    if(Gait_mode<=1):
                        L_leg_p[0:3, 0] = (Rlfoot_pos[3:6, 0] + (leg_FL_homing[0:3,0] + leg_RL_homing[0:3,0])/2)
                        R_leg_p[0:3, 0] = (Rlfoot_pos[0:3, 0] + (leg_FR_homing[0:3,0] + leg_RR_homing[0:3,0])/2)
                    else:
                        L_leg_p[0:3, 0] = (Rlfoot_pos[3:6, 0] + (leg_FL_homing[0:3,0] + leg_FR_homing[0:3,0])/2)
                        R_leg_p[0:3, 0] = (Rlfoot_pos[0:3, 0] + (leg_RL_homing[0:3,0] + leg_RR_homing[0:3,0])/2)
                else:
                    Homing_height_t = (Gait_home_height - des_basex[2]) * math.sin(tx / t_homing / 2. * math.pi)
                    Homing_height_velt = (1/t_homing/2*math.pi) * (Gait_home_height - des_basex[2]) * math.cos(
                        tx/t_homing/2. * math.pi)
                    Homing_height_acct = -(1/t_homing/2*math.pi)**2 * (Gait_home_height - des_basex[2]) * math.sin(
                        tx/t_homing/2. * math.pi)
                    des_base[2] = des_basex[2] + Homing_height_t
                    des_base_vel[2] = Homing_height_velt


                    FRleg_y_offset = (Gait_func.FR_leg_offset_y + Gait_func.FR_thigh_y)
                    homing_FR_y_offset = (FRleg_y_offset - (leg_FR_homingx[1,0] - des_base[1])) * math.sin(tx / t_homing / 2. * math.pi)
                    des_FR_p[1,0] = homing_FR_y_offset + (leg_FR_homingx[1,0])
                    des_FR_v[1,0] = (FRleg_y_offset - (leg_FR_homingx[1,0] - des_base[1])) * (1/t_homing/2*math.pi) * math.cos(tx / t_homing / 2. * math.pi)

                    FLleg_y_offset = (Gait_func.FL_leg_offset_y + Gait_func.FL_thigh_y)
                    homing_FL_y_offset = (FLleg_y_offset - (leg_FL_homingx[1, 0] - des_base[1])) * math.sin(
                        tx / t_homing / 2. * math.pi)
                    des_FL_p[1, 0] = homing_FL_y_offset + (leg_FL_homingx[1, 0])
                    des_FL_v[1, 0] = (FLleg_y_offset - (leg_FL_homingx[1, 0] - des_base[1])) * (1/t_homing/2*math.pi) * math.cos(tx / t_homing / 2. * math.pi)

                    RRleg_y_offset = (Gait_func.RR_leg_offset_y + Gait_func.RR_thigh_y)
                    homing_RR_y_offset = (RRleg_y_offset - (leg_RR_homingx[1,0] - des_base[1])) * math.sin(tx / t_homing / 2. * math.pi)
                    des_RR_p[1,0] = homing_RR_y_offset + (leg_RR_homingx[1,0])
                    des_RR_v[1,0] = (RRleg_y_offset - (leg_RR_homingx[1,0] - des_base[1])) * (1/t_homing/2*math.pi) * math.cos(tx / t_homing / 2. * math.pi)

                    RLleg_y_offset = (Gait_func.RL_leg_offset_y + Gait_func.RL_thigh_y)
                    homing_RL_y_offset = (RLleg_y_offset - (leg_RL_homingx[1, 0] - des_base[1])) * math.sin(
                        tx / t_homing / 2. * math.pi)
                    des_RL_p[1, 0] = homing_RL_y_offset + (leg_RL_homingx[1, 0])
                    des_RL_v[1, 0] = (RLleg_y_offset - (leg_RL_homingx[1, 0] - des_base[1])) * (1/t_homing/2*math.pi) * math.cos(tx / t_homing / 2. * math.pi)

                    ################################# be careful that the base location is non the same with link[-1] state, we use baseposition as CoM position
                    # des_base_fix = pybullet.getBasePositionAndOrientation(go1id)[0]
                    # des_base_fix = copy.deepcopy(des_base)
                    # leg_FR_homing[:, 0] = (links_pos[State_estimator.FR_soleid, :]).T
                    # leg_FL_homing[:, 0] = (links_pos[State_estimator.FL_soleid, :]).T
                    # leg_RR_homing[:, 0] = (links_pos[State_estimator.RR_soleid, :]).T
                    # leg_RL_homing[:, 0] = (links_pos[State_estimator.RL_soleid, :]).T
                    leg_FR_homing = copy.deepcopy(des_FR_p)
                    leg_FL_homing = copy.deepcopy(des_FL_p)
                    leg_RR_homing = copy.deepcopy(des_RR_p)
                    leg_RL_homing = copy.deepcopy(des_RL_p)

                    des_base_fix = copy.deepcopy(des_base)

                    ################# Force_distribution############
                    right_leg_support = 2 ###double support
                    base_p = np.zeros([3,1])
                    base_p[:,0] = des_base.T
                    cop_xyz = base_p
                    base_acc = np.zeros([3,1])
                    base_acc[2,0] = Homing_height_acct

                    body_R_des = np.zeros([3, 1])
                    body_Rv_des = np.zeros([3, 1])
                    base_R_acc = np.zeros([3, 1])
                    R_leg_p = np.zeros([3, 1])
                    L_leg_p = np.zeros([3, 1])
                    L_leg_p[0:3, 0] = ((leg_FL_homing[0:3,0] + leg_RL_homing[0:3,0])/2)
                    R_leg_p[0:3, 0] = ((leg_FR_homing[0:3,0] + leg_RR_homing[0:3,0])/2)
                    # print("des_base_fix:", des_base)

                ###### IK-based controller-CoM pos and ori control
                for j in range(0, 3):
                    body_p_mea[j,0] = bas_pos[j] - L_leg_p_mea[j,0]
                    body_v_mea[j,0] = bas_posv[j]
                    body_p_ref[j,0] = des_base[j] - L_leg_p[j,0]
                    body_v_ref[j, 0] = des_base_vel[j]

                    body_r_mea[j] = base_R[j]
                    body_rv_mea[j] = bas_oriv[j]
                    body_r_ref[j] = body_R_des[j]
                    body_rv_ref[j] = body_Rv_des[j]

                det_base_p, det_base_R = State_estimator.CoM_pos_fb_tracking(body_p_ref, body_v_ref,body_p_mea,
                                                                             body_v_mea,body_r_ref, body_rv_ref,body_r_mea, body_rv_mea)
                if(tx<t_homing):
                    pd_lamda = pow(tx/t_homing,2)
                else:
                    pd_lamda = 1

                for j in range(0, 3):
                    des_base[j] = copy.deepcopy(base_p[j,0]) + pd_lamda * det_base_p[j,0]
                    body_R_des[j] += pd_lamda * det_base_R[j,0]

                ########### Ik ###############################
                ### using the reference Jacobian computed by desired angles
                q_FR, J_FR = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max=15, lamda=0.55)
                q_FL, J_FL = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
                q_RR, J_RR = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max=15, lamda=0.55)
                q_RL, J_RL = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)
                ### using the Jacobian computed by measured angles
                # q_FR, xx = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max =15, lamda=0.55)
                # q_FL, xx = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
                # q_RR, xx = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max =15, lamda=0.55)
                # q_RL, xx = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)
                Q_cmd[0:3,i-1] = q_FR[0:3]
                Q_cmd[3:6,i-1] = q_FL[0:3]
                Q_cmd[6:9,i-1] = q_RR[0:3]
                Q_cmd[9:12,i-1] = q_RL[0:3]
                q_cmd = Q_cmd[:,i-1]
                q_vel_cmd = (q_cmd -q_cmd_pre)/dt

                Pos_ref[0:3, i - 1] = des_base[0:3]
                Pos_ref[3:6, i - 1] = body_R_des[0:3, 0]
                Pos_ref[6:9, i - 1] = des_FR_p[0:3, 0]
                Pos_ref[9:12, i - 1] = des_FL_p[0:3, 0]
                Pos_ref[12:15, i - 1] = des_RR_p[0:3, 0]
                Pos_ref[15:18, i - 1] = des_RL_p[0:3, 0]

                RLfoot_com_pos[:,i-1] = Rlfoot_pos[:,0]
                outx[i-1,0:12] = res_outx
                outx[i-1,12:15] = (com_intex[0:3,0]).T
                #### force distribution #############################
                #### initial guess
                F_tot, Force_FR,Force_FL,Force_RR,Force_RL = Grf_computer.Grf_ref_pre(Gait_mode,right_leg_support,base_p,base_acc,
                                                                                      body_R_des,base_R_acc,des_FR_p,des_FL_p,
                                                                                      des_RR_p,des_RL_p,R_leg_p,L_leg_p,cop_xyz)

                Grf_ref[0:6,i-1] = F_tot[:,0]
                Grf_ref[6:9,i-1] = Force_FR[0:3,0]
                Grf_ref[9:12,i-1] = Force_FL[0:3,0]
                Grf_ref[12:15,i-1] = Force_RR[0:3,0]
                Grf_ref[15:18,i-1] = Force_RL[0:3,0]

                ###### optimization
                F_leg_opt = Grf_computer.Grf_ref_opt(Gait_mode,right_leg_support,base_p,des_FR_p,des_FL_p,des_RR_p,des_RL_p)
                Force_FR = F_leg_opt[0:3,:]
                Force_FL = F_leg_opt[3:6,:]
                Force_RR = F_leg_opt[6:9,:]
                Force_RL = F_leg_opt[9:12,:]
                # print(F_leg_opt)
                Grf_opt[:,i-1] = F_leg_opt[:,0]

                ############### Force controller ####################################
                # for j in range(0, 3):
                #     Relative_FR_pos[j] = des_FR_p[j, 0] - des_base[j]
                #     Relative_FR_vel[j] = des_FR_v[j, 0] - des_base_vel[j]
                #     Relative_FL_pos[j] = des_FL_p[j, 0] - des_base[j]
                #     Relative_FL_vel[j] = des_FL_v[j, 0] - des_base_vel[j]
                #     Relative_RR_pos[j] = des_RR_p[j, 0] - des_base[j]
                #     Relative_RR_vel[j] = des_RR_v[j, 0] - des_base_vel[j]
                #     Relative_RL_pos[j] = des_RL_p[j, 0] - des_base[j]
                #     Relative_RL_vel[j] = des_RL_v[j, 0] - des_base_vel[j]
                #
                # torque_FR,torque_FR_fb = Force_controller.torque_cmd(FR_support,J_FR, Force_FR,q_cmd[0:3], q_mea[0:3], q_vel_cmd[0:3], dq_mea[0:3],
                #                                         Relative_FR_pos, Relative_FR_pos_mea, Relative_FR_vel, Relative_FR_vel_mea,Gravity_comp[0:3])
                # torque_FL,torque_FL_fb = Force_controller.torque_cmd(FL_support,J_FL, Force_FL,q_cmd[3:6], q_mea[3:6], q_vel_cmd[3:6], dq_mea[3:6],
                #                                         Relative_FL_pos, Relative_FL_pos_mea, Relative_FL_vel, Relative_FL_vel_mea,Gravity_comp[3:6])
                # torque_RR,torque_RR_fb = Force_controller.torque_cmd(RR_support,J_RR, Force_RR,q_cmd[6:9], q_mea[6:9], q_vel_cmd[6:9], dq_mea[6:9],
                #                                         Relative_RR_pos, Relative_RR_pos_mea, Relative_RR_vel, Relative_RR_vel_mea,Gravity_comp[6:9])
                # torque_RL,torque_RL_fb = Force_controller.torque_cmd(FL_support,J_RL, Force_RL,q_cmd[9:12], q_mea[9:12], q_vel_cmd[9:12], dq_mea[9:12],
                #                                         Relative_RL_pos, Relative_RL_pos_mea, Relative_RL_vel, Relative_RL_vel_mea,Gravity_comp[9:12])
                # # ################################### send torque command #################################
                # Torque_cmd[0:3, i - 1] = torque_FR[0:3]
                # Torque_cmd[3:6, i - 1] = torque_FL[0:3]
                # Torque_cmd[6:9, i - 1] = torque_RR[0:3]
                # Torque_cmd[9:12, i - 1] = torque_RL[0:3]
                #
                # Torque_FB[0:3, i - 1] = torque_FR_fb[0:3]
                # Torque_FB[3:6, i - 1] = torque_FL_fb[0:3]
                # Torque_FB[6:9, i - 1] = torque_RR_fb[0:3]
                # Torque_FB[9:12, i - 1] = torque_RL_fb[0:3]
                #
                # torque_cmd = Torque_cmd[:, i - 1]
                # urobtx.setActuatedJointTorques(torque_cmd)
                urobtx.setActuatedJointPositions(Q_cmd[:, i - 1])


    Force_FR_old = copy.deepcopy(Force_FR)
    Force_FL_old = copy.deepcopy(Force_FL)
    Force_RR_old = copy.deepcopy(Force_RR)
    Force_RL_old = copy.deepcopy(Force_RL)
    links_pos_prev = copy.deepcopy(links_pos)
    links_vel_prev = copy.deepcopy(links_vel)
    q_cmd_pre = copy.deepcopy(q_cmd)
    body_r_mea_pre = copy.deepcopy(body_r_mea)


    if (i==i_total):  ### save data

        fsr_mea_dir = mesh_dir + '/go1/go1_measure_fsr.txt'
        angle_ref_dir = mesh_dir + '/go1/go1_angle_ref.txt'
        angle_mea_dir = mesh_dir + '/go1/go1_angle_mea.txt'
        torque_ref_dir = mesh_dir + '/go1/go1_torque_ref.txt'
        torque_mea_dir = mesh_dir + '/go1/go1_torque_fb.txt'
        pos_ref_dir = mesh_dir + '/go1/go1_pos_ref.txt'
        grf_ref_dir = mesh_dir + '/go1/go1_grf_ref.txt'
        grf_opt_dir = mesh_dir + '/go1/go1_grf_opt.txt'
        kmp_trajectory_dir = mesh_dir + '/go1/go1_kmp_trajectory.txt'
        nlp_trajectory_dir = mesh_dir + '/go1/go1_nlp_trajectory.txt'
        state_mea_dir = mesh_dir + '/go1/go1_state_mea.txt'

        np.savetxt(fsr_mea_dir, FSR_measured,
                   fmt='%s', newline='\n')
        np.savetxt(angle_ref_dir, Q_cmd, fmt='%s',
                   newline='\n')
        np.savetxt(angle_mea_dir, Q_measure, fmt='%s',
                   newline='\n')
        np.savetxt(torque_ref_dir, Torque_cmd, fmt='%s',
                   newline='\n')
        np.savetxt(torque_mea_dir, Torque_FB, fmt='%s',
                   newline='\n')
        np.savetxt(pos_ref_dir, Pos_ref, fmt='%s',
                   newline='\n')
        np.savetxt(grf_ref_dir, Grf_ref, fmt='%s',
                   newline='\n')
        np.savetxt(grf_opt_dir, Grf_opt, fmt='%s',
                   newline='\n')
        np.savetxt(kmp_trajectory_dir, RLfoot_com_pos, fmt='%s',
                   newline='\n')
        np.savetxt(nlp_trajectory_dir, outx, fmt='%s',
                   newline='\n')
        np.savetxt(state_mea_dir, state_feedback, fmt='%s',
                   newline='\n')

    i += 1

    ##### doesn't use it in realtime simu mode
    # pybullet.stepSimulation()

    sim_env.step()

    # ##############only work in real-time model???? a bug #########################3
    # ls = pybullet.getLinkState(go1id, torso1_linkid)
    # # print("torso_link:",ls[0])
    # if ((hasPrevPose) and (i - n_t_homing >300) and (i % 3 ==0)):
    #     # pybullet.addUserDebugLine(prevPose, lsx, [0, 0, 0.3], 1, trailDuration)
    #     sim_env.addDebugLine(prevPose1, ls[4])
    # # prevPose = lsx_ori
    # prevPose1 = ls[4]
    # hasPrevPose = 1
