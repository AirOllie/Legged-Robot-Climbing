#### python envir setup
from __future__ import print_function

import os
from os.path import dirname, join, abspath
import sys

from pathlib import Path


### pinocchio
import pinocchio as pin
from pinocchio.explog import log
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from pino_robot_ik import CLIK                        #### IK solver
from robot_tracking_controller import Gait_Controller #### controller
from LIP_motion_planner import Gait                   #### Gait planner

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


#################################################################################################
global base_homing
global pr_homing
global pl_homing

############################################################################### robot setup ###############
########whold-body simu:
Full_body_simu = True
##########for robot with float-base ################################
Freebase = True

mesh_dir = str(Path(__file__).parent.absolute())

# You should change here to set up your own URDF file
if Full_body_simu:
    # urdf_filename = mesh_dir + '/go1_description/urdf/go1_full_no_grippers.urdf'
    urdf_filename = mesh_dir + '/go1_description/urdf/go1.urdf'
else:
    urdf_filename = mesh_dir + '/go1_description/urdf/go1.urdf'

### pinocchio load urdf
if Freebase:
    robot =  RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir, pin.JointModelFreeFlyer())
    addFreeFlyerJointLimits(robot)
else:
    robot = RobotWrapper.BuildFromURDF(urdf_filename, mesh_dir)
## explore the model class
for name, function in robot.model.__class__.__dict__.items():
    print(' **** %s: %s' % (name, function.__doc__))
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$444' )
print('standard model: dim=' + str(len(robot.model.joints)))
for jn in robot.model.joints:
    print(jn)
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
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pinocchio load urdf finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")

################################################### pybullet simulation loading ###########################
### intial pose for go1 in pybullet
sim_rate = 200
dt = 1./sim_rate
sim_env = SimEnv(sim_rate=sim_rate)

urobtx = SimRobot(urdfFileName=urdf_filename,
                 basePosition=[0, 0, 0.446],
                 baseRPY=[0, 0, 0])
go1id = urobtx.id


num_joints = urobtx.getNumJoints()
num_actuated_joint = urobtx.getNumActuatedJoints()
actuation_joint_index = urobtx.getActuatedJointIndexes()
# print("actuated",num_actuated_joint)
# print("actuated joint index",actuation_joint_index)

Homing_pose = np.zeros(num_actuated_joint)
### Homing_pose: four legs:FR, FL, RR, RL####### is important for walking in place
Homing_pose[0] =  0
Homing_pose[1] =  0.75
Homing_pose[2] =  -1.5
Homing_pose[3] = 0

Homing_pose[4] = 0
Homing_pose[5] =  0.75
Homing_pose[6] =  -1.5
Homing_pose[7] = 0

Homing_pose[8] = 0
Homing_pose[9] =  0.75
Homing_pose[10] =  -1.5
Homing_pose[11] =  0

Homing_pose[12] =  0
Homing_pose[13] =  0.75
Homing_pose[14] =  -1.5
Homing_pose[15] =  0

print("Homing_pose:",Homing_pose)


# q_initial = np.zeros(num_actuated_joint)
# for i in range(0, num_actuated_joint):
#     q_initial[i] = Homing_pose[i]*0.01
# urobtx.resetJointStates(q_initial)

q_cmd = np.zeros(num_actuated_joint)
torque_cmd = np.zeros(num_actuated_joint)
t_homing = 5
n_t_homing = round(t_homing/dt)

useRealTimeSimulation = 0
pybullet.setRealTimeSimulation(useRealTimeSimulation)
############################## enable FSR sensoring
idFR_fsr = idFR[2]+1
idFL_fsr = idFL[2]+1
idRR_fsr = idRR[2]+1
idRL_fsr = idRL[2]+1
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idFR_fsr,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idFL_fsr,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idRR_fsr,enableSensor=1)
pybullet.enableJointForceTorqueSensor(bodyUniqueId=go1id,jointIndex=idRL_fsr,enableSensor=1)

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!pybullet load environment finishing!!!!!!!!!!!!!!!!!!!!!!!!!!")
######################################################3
##### Gait_Controller
Controller_ver = Gait_Controller(urbodx = urobtx, id = go1id,verbose=True)

trailDuration = 10
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0.446]
hasPrevPose = 0

t=0.
leg_FL_homing_fix = []
leg_FR_homing_fix = []
leg_RL_homing_fix = []
leg_RR_homing_fix = []
base_home_fix = []

torso1_linkid = 0
left_sole_linkid = 36
right_sole_linkid = 42



mesh_dirx = str(Path(__file__).parent.absolute())
angle_cmd_filename = mesh_dirx + 'go1/go1_angle_nmpc.txt'
torque_cmd_filename = mesh_dirx + 'go1/go1_ref_torque_nmpc.txt'

joint_angle_cmd = np.loadtxt(angle_cmd_filename)  ### linux/ubuntu
joint_torque_cmd = np.loadtxt(torque_cmd_filename)  ### linux/ubuntu

print(type(joint_angle_cmd))
row_num = joint_angle_cmd.shape[0]
col_num = joint_angle_cmd.shape[1]
print(col_num)


##############For kinematics;
des_FR_p = np.zeros([3,1])
des_FL_p = np.zeros([3,1])
des_RR_p = np.zeros([3,1])
des_RL_p = np.zeros([3,1])

### initialization for sim_robots, only once for FK
des_FL = np.array([3,1,1])
oMdes_FL = pin.SE3(np.eye(3), des_FL)
JOINT_ID_FL = idFL[-1]
#########leg IK ###############3
IK_leg = CLIK(robot, oMdes_FL, JOINT_ID_FL, Freebase)

###########gait planner
Gait_func = Gait(T=0.8, Sx = 0.05, Sy = 0, Sz = 0, lift = 0.02, T_num = 20, Dsp_ratio=0.1, dt = dt, Qudrupedal=True)

N_gait = Gait_func.step_location()
print(N_gait)
N_gait += n_t_homing
print(N_gait)


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

####################################### main loop for robot gait generation and control ####################
i = 0

while i<=N_gait:
    if (useRealTimeSimulation):
        t = t + dt
    else:
        t = t + dt

    ##########============= read data from an offline file ==========
    # k_order = i
    #
    # # for jx in range(3):
    # #     k_order_row = 2 * k_order
    # #     k_order_col = 4*jx+1
    # #     q_cmd[jx] = joint_angle_cmd[k_order_row,k_order_col]
    # #     torque_cmd[jx] = 2*joint_torque_cmd[k_order_row,k_order_col]
    #
    # for jx in range(3,6):
    #     k_order_row = 2 * k_order
    #     k_order_col = 4*jx+1
    #     q_cmd[jx] = joint_angle_cmd[k_order_row,k_order_col]
    #     torque_cmd[jx] = 2*joint_torque_cmd[k_order_row,k_order_col]
    #
    # torque_cmd;
    #
    # print("q_cmd:", q_cmd)
    #
    # # urobtx.setActuatedJointPositions(q_cmd)
    # urobtx.setActuatedJointTorques(torque_cmd)

    xx_joint = urobtx.getActuatedJointPositions() ### catersian position, not angles
    ############## robot kinematic control loop##################################################
    if i<=n_t_homing:            ############# initial pose

        ### starting from neutral states: the robot would face jerks
        if (i<=10):
            # pybullet.applyExternalForce(go1id,0,forceObj=[100,0,0],posObj=[0,0,0],flags=1)
            pybullet. applyExternalTorque(go1id, 0, torqueObj=[0, -10, 0], flags=1)
        Homing_pose_t = Homing_pose*math.sin(t/t_homing/2.*math.pi)
        Homing_vel_t  = (1/t_homing/2.*math.pi) * Homing_pose*math.cos(t/t_homing/2.*math.pi)
        # Homing_pose_t =  (t/t_homing)**2 * Homing_pose+ (1-(t/t_homing)**2) * xx_joint   ###

        # for pinocchio
        q = robot.q0 # base joint, FL, FR, RL, RR
        q[0 + 7:4 + 7] = Homing_pose_t[4:8]
        q[4 + 7:8 + 7] = Homing_pose_t[0:4]
        q[8 + 7:12 + 7] = Homing_pose_t[12:16]
        q[12 + 7:16 + 7]= Homing_pose_t[8:12]

        joint_opt[i] = Homing_pose_t
        urobtx.setActuatedJointPositions(Homing_pose_t)

        # acutated_joint_pos, acutated_joint_vel = Controller_ver.get_acutated_joint_pos_vel()
        # T_cmd = 0.5*(Homing_pose_t - acutated_joint_pos) + 0.1 * (Homing_vel_t - acutated_joint_vel)
        # urobtx.setActuatedJointTorques(T_cmd)



        ####### state estimation ###################################################################
        base_pos_m = pybullet.getBasePositionAndOrientation(go1id)[0]

        if Freebase:
            robot.model.jointPlacements[1] = pin.SE3(np.eye(3), np.array(base_pos_m))

        pin.forwardKinematics(robot.model,robot.data,q)
        leg_FL_homing = robot.data.oMi[idFL[-1]].translation
        leg_FR_homing = robot.data.oMi[idFR[-1]].translation
        leg_RL_homing = robot.data.oMi[idRL[-1]].translation
        leg_RR_homing = robot.data.oMi[idRR[-1]].translation

        base_home_fix = base_pos_m
        leg_FL_homing_fix = tuple(leg_FL_homing)
        leg_FR_homing_fix = tuple(leg_FR_homing)
        leg_RL_homing_fix = tuple(leg_RL_homing)
        leg_RR_homing_fix = tuple(leg_RR_homing)

        leg_FL_homing_fix1 = tuple(leg_FL_homing)
        leg_FR_homing_fix1 = tuple(leg_FR_homing)
        leg_RL_homing_fix1 = tuple(leg_RL_homing)
        leg_RR_homing_fix1 = tuple(leg_RR_homing)

        # #### A
        q_FR = Homing_pose_t[0:3]
        q_FL = Homing_pose_t[4:7]
        q_RR = Homing_pose_t[8:11]
        q_RL = Homing_pose_t[12:15]
        des_FR_p[0:3,0] = leg_FR_homing
        des_FL_p[0:3,0] = leg_FL_homing
        des_RR_p[0:3,0] = leg_RR_homing
        des_RL_p[0:3,0] = leg_RL_homing
        h_com = base_pos_m[2] - (leg_FR_homing[2]+leg_FL_homing[2]+leg_RR_homing[2]+leg_RL_homing[2])/4

        q_ik = Homing_pose_t



        links_pos, links_vel, links_acc = Controller_ver.get_link_vel_vol(i,dt,links_pos_prev,links_vel_prev)
        links_pos_prev = links_pos
        links_vel_prev = links_vel

        support_flag[i] = 0  ### 0, double support, right support
        # ###################state feedback ###########################################
        # gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m,links_pos, links_vel, links_acc = \
        # Controller_ver.state_estimation(i,dt,support_flag,links_pos_prev,links_vel_prev,gcom_pre)
        #
        # state_feedback[i,0:3] = gcom_m
        # state_feedback[i, 3:6] = right_sole_pos
        # state_feedback[i, 6:9] = left_sole_pos
        # state_feedback[i, 9:15] = right_ankle_force
        # state_feedback[i, 15:21] = left_ankle_force
        # state_feedback[i, 21:24] = gcop_m
        # state_feedback[i, 24:27] = dcm_pos_m
        # state_feedback[i, 27:30] = com_vel_m
        # state_feedback[i, 30:33] = base_pos_m
        # state_feedback[i, 33:36] = base_angle_m
        #
        # links_pos_prev = links_pos
        # links_vel_prev = links_vel
        # gcom_pre = gcom_m
        # com_feedback_base = gcom_m
        # com_ref_base = base_pos_m
    else:
        ####



        ##########============= read data from an offline file =
        # k_order = i - round(t_homing/dt)
        #
        # for jx in range(num_actuated_joint):
        #     k_order_row = 2 * k_order
        #     k_order_col = 4*jx+1
        #     q_cmd[jx] = joint_angle_cmd[k_order_row,k_order_col]
        #     torque_cmd[jx] = joint_torque_cmd[k_order_row,k_order_col]
        #
        # print("q_cmd:", q_cmd)
        #
        # # urobtx.setActuatedJointPositions(q_cmd)
        # urobtx.setActuatedJointTorques(torque_cmd)


        # ######## reference trajectory generation #############################
        if Freebase: #routine1: change the base position and orientation for pinocchio IK: time-cost process due to the redundant freedom
            ################# test: data format is used for pinocchio
            des_base = np.array([0,
                                 0 * (math.sin((t - t_homing) * 50 * math.pi / 180)),
                                 -0.03 * abs(math.sin((t - t_homing) * 50 * math.pi / 180))]) + np.array(base_home_fix)
            # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_base)
            # des_FL = np.array(leg_FL_homing_fix)
            # oMdes_FL = pin.SE3(np.eye(3), des_FL)
            # des_FR = np.array(leg_FR_homing_fix)
            # oMdes_FR = pin.SE3(np.eye(3), des_FR)
            # des_RL = np.array(leg_RL_homing_fix)
            # oMdes_RL = pin.SE3(np.eye(3), des_RL)
            # des_RR = np.array(leg_RR_homing_fix)
            # oMdes_RR = pin.SE3(np.eye(3), des_RR)

        # ## gait planning-based on LIP
        j_index = int(i - n_t_homing)
        # pos_base = Gait_func.Ref_com_lip_update(j_index,h_com)
        #
        # Rlfoot_pos = Gait_func.FootpR(j_index)
        #
        #
        # des_base = np.array(base_home_fix)
        # des_base[0] = base_home_fix[0] + pos_base[0,0]
        # des_base[1] = base_home_fix[1] + pos_base[1,0]*0.00
        # des_base[2] = base_home_fix[2] + pos_base[2,0]
        # #### pair:FR--RL-; pair:FL--RR
        # des_FR_p[0:3,0] = leg_FR_homing + Rlfoot_pos[0:3, 0]
        # des_RL_p[0:3,0] = leg_RL_homing + Rlfoot_pos[0:3, 0]
        # des_FL_p[0:3, 0] = leg_FL_homing + Rlfoot_pos[3:6, 0]
        # des_RR_p[0:3, 0] = leg_RR_homing + Rlfoot_pos[3:6, 0]

        body_R_des = np.zeros([3, 1])

        q_FR, J_FR = IK_leg.ik_close_form(des_base, body_R_des, des_FR_p, q_FR, 0, It_max =15, lamda=0.55)
        q_FL, J_FL = IK_leg.ik_close_form(des_base, body_R_des, des_FL_p, q_FL, 1, It_max=15, lamda=0.55)
        q_RR, J_RR = IK_leg.ik_close_form(des_base, body_R_des, des_RR_p, q_RR, 2, It_max =15, lamda=0.55)
        q_RL, J_RL = IK_leg.ik_close_form(des_base, body_R_des, des_RL_p, q_RL, 3, It_max=15, lamda=0.55)

        ###

        q_ik[0:3] = q_FR
        q_ik[4:7] = q_FL
        q_ik[8:11] = q_RR
        q_ik[12:15] = q_RL

        urobtx.setActuatedJointPositions(q_ik)

        # urobtx.setActuatedJointTorques(q_ik)



        if(j_index==(N_gait-n_t_homing)):
            plt.figure()
            plt.subplot(3, 2, 1)
            plt.plot(Gait_func.comx)
            plt.plot(Gait_func.px)
            plt.subplot(3, 2, 2)
            plt.plot(Gait_func.comy)
            plt.plot(Gait_func.py)
            plt.subplot(3, 2, 3)
            plt.plot(Gait_func.comvx)
            plt.subplot(3, 2, 4)
            plt.plot(Gait_func.comvy)
            plt.subplot(3, 2, 5)
            plt.plot(Gait_func.comax)
            plt.subplot(3, 2, 6)
            plt.plot(Gait_func.comay)
            plt.show()

            plt.figure()
            plt.subplot(2, 3, 1)
            plt.plot(Gait_func._Rfootx)
            plt.plot(Gait_func._Lfootx)
            plt.subplot(2, 3, 2)
            plt.plot(Gait_func._Rfooty)
            plt.plot(Gait_func._Lfooty)
            plt.subplot(2, 3, 3)
            plt.plot(Gait_func._Rfootz)
            plt.plot(Gait_func._Lfootz)
            plt.subplot(2, 3, 4)
            plt.plot(Gait_func._Rfootvx)
            plt.plot(Gait_func._Lfootvx)
            plt.subplot(2, 3, 5)
            plt.plot(Gait_func._Rfootvy)
            plt.plot(Gait_func._Lfootvy)
            plt.subplot(2, 3, 6)
            plt.plot(Gait_func._Rfootvz)
            plt.plot(Gait_func._Lfootvz)
            plt.show()

        # ############ IK-solution for the float-based humanod: providing initial guess "homing_pose" #######################
        # ########### set endeffector id for ik using pinocchio
        # JOINT_ID_FL = idFL[-1]
        # JOINT_ID_FR = idFR[-1]
        # JOINT_ID_RL = idRL[-1]
        # JOINT_ID_RR = idRR[-1]
        #
        # q_ik = joint_lower_leg_ik(robot, oMdes_FL, JOINT_ID_FL, oMdes_FR, JOINT_ID_FR, oMdes_RL, JOINT_ID_RL, oMdes_RR, JOINT_ID_RR, Freebase, Homing_pose)
        #
        # if (i % 10 == 0):
        #     print("q_ik_cmd:", q_ik)
        # ######## joint command: position control mode ###########################
        # joint_opt[i] = q_ik
        # urobtx.setActuatedJointPositions(q_ik)



        # ###################state feedback ###########################################
        # gcom_m, right_sole_pos, left_sole_pos, base_pos_m, base_angle_m, right_ankle_force, left_ankle_force, gcop_m, support_flag, dcm_pos_m, com_vel_m,links_pos, links_vel, links_acc = \
        # Controller_ver.state_estimation(i,dt,support_flag,links_pos_prev,links_vel_prev,gcom_pre)
        #
        # state_feedback[i,0:3] = gcom_m
        # state_feedback[i, 3:6] = right_sole_pos
        # state_feedback[i, 6:9] = left_sole_pos
        # state_feedback[i, 9:15] = right_ankle_force
        # state_feedback[i, 15:21] = left_ankle_force
        # state_feedback[i, 21:24] = gcop_m
        # state_feedback[i, 24:27] = dcm_pos_m
        # state_feedback[i, 27:30] = com_vel_m
        # state_feedback[i, 30:33] = base_pos_m
        # state_feedback[i, 33:36] = base_angle_m
        #
        # links_pos_prev = links_pos
        # links_vel_prev = links_vel
        # gcom_pre = gcom_m
        #
        # if ((abs(base_angle_m[0]) >=20* math.pi / 180) or (abs(base_angle_m[1]) >=20* math.pi / 180) ): ### falling down
        #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_traj_nmpc.txt', traj_opt,fmt='%s', newline='\n')
        #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_angle_nmpc.txt', joint_opt,fmt='%s', newline='\n')
        #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_state_est_nmpc.txt', state_feedback,fmt='%s', newline='\n')
        #
        # ################## IK-based control: in this case, we can use admittance control, preview control and PD controller for CoM control #################################33
        # com_ref_det = np.array(des_base) - np.array(com_ref_base)
        # com_feedback_det = np.array(gcom_m) - np.array(com_feedback_base)
        # angle_ref_det = des_base_ori
        # angle_feedback_det = base_angle_m
        #
        # det_comxxxx, det_body_anglexxxx =Controller_ver.CoM_Body_pd(dt,com_ref_det, com_feedback_det, com_ref_det_pre, com_feedback_det_pre, angle_ref_det,angle_feedback_det, angle_ref_det_pre, angle_feedback_det_pre)
        # des_com_pos_control = det_comxxxx + np.array(des_base)
        # det_base_angle_control = det_body_anglexxxx
        # det_base_matrix_control = Controller_ver.RotMatrixfromEuler(det_base_angle_control)
        # robot.model.jointPlacements[1] = pin.SE3(det_base_matrix_control, des_com_pos_control)
        # # robot.model.jointPlacements[1] = pin.SE3(np.eye(3), des_com_pos_control)
        #
        # com_ref_det_pre = com_ref_det
        # com_feedback_det_pre = com_feedback_det
        # angle_ref_det_pre = angle_ref_det
        # angle_feedback_det_pre = angle_feedback_det
        #

        # ###########################===========================================================
        # ######################################################################################


    i += 1
    # if (i==FileLength-1):
    #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_traj_nmpc.txt',traj_opt,fmt='%s',newline='\n')
    #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_angle_nmpc.txt', joint_opt,fmt='%s', newline='\n')
    #     np.savetxt('/home/jiatao/anaconda3/envs/nameOfEnv/pybullet_gym/go1/go1_state_est_nmpc.txt', state_feedback,fmt='%s', newline='\n')
    #
    ##### doesn't use it in realtime simu mode
    #pybullet.stepSimulation()
    # time.sleep(dt)
    sim_env.step()

    ##############only work in real-time model???? a bug #########################3
    # ls = pybullet.getLinkState(go1id, torso1_linkid)
    # print("torso_link:",ls[0])
    # if (hasPrevPose):
    #     # pybullet.addUserDebugLine(prevPose, lsx, [0, 0, 0.3], 1, trailDuration)
    #     pybullet.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    # # prevPose = lsx_ori
    # prevPose1 = ls[4]
    # hasPrevPose = 1
