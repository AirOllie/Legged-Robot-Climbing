import os
import numpy as np
import math
import pybullet


class Gait_Controller:
    def __init__(self, urbodx, id, gait_mode,verbose=True):
        self.robot = urbodx
        self.id = id
        self.mass = urbodx.getRobotMass()
        self.jointnumber = urbodx.getNumJoints()  #####note that the jointnumber equals to the linkname when the base_link is not count
        self.jointactuatednumber = urbodx.getNumActuatedJoints()
        self.jointactuatedindex = urbodx.getActuatedJointIndexes()
        self.Link_name_index_dict = urbodx.getLinkIndexNameMap()
        self.g = 9.8

        ######
        self.kpp = 0.2
        self.kdp = 0.0055
        self.kppx = 0.1
        self.kdpx = 0.005
        self.kpr = 0.125
        self.kdr = 0.005
        if(gait_mode==0):
            self.kpp = 0.2
            self.kdp = 0.0055
            self.kppx = 0.1
            self.kdpx = 0.005
            self.kpr = 0.125
            self.kdr = 0.005
        else:
            self.kpp = 0.2
            self.kdp = 0.0055
            self.kppx = 0.1
            self.kdpx = 0.005
            self.kpr = 0.125
            self.kdr = 0.005


        if verbose:
            print('*' * 100 + '\nPyBullet Controller Info ' + '\u2193 ' * 20 + '\n' + '*' * 100)
            print('robot ID:              ', id)
            xxx = self.robot.id
            print('urbodx.ID:              ', xxx)
            print('robot mass:              ', self.mass)
            print('*' * 100 + '\nPyBullet Controller Info ' + '\u2191 ' * 20 + '\n' + '*' * 100)

        # self.FR_soleid = self.jointactuatedindex[3]+1
        # self.FL_soleid = self.jointactuatedindex[7]+1
        # self.RR_soleid = self.jointactuatedindex[11]+1
        # self.RL_soleid = self.jointactuatedindex[15]+1
        ### for origin_urdf file
        self.FR_soleid = self.jointactuatedindex[2]+1
        self.FL_soleid = self.jointactuatedindex[5]+1
        self.RR_soleid = self.jointactuatedindex[8]+1
        self.RL_soleid = self.jointactuatedindex[11]+1

        pybullet.enableJointForceTorqueSensor(bodyUniqueId=self.id, jointIndex=self.FR_soleid, enableSensor=1)
        pybullet.enableJointForceTorqueSensor(bodyUniqueId=self.id, jointIndex=self.FL_soleid, enableSensor=1)
        pybullet.enableJointForceTorqueSensor(bodyUniqueId=self.id, jointIndex=self.RR_soleid, enableSensor=1)
        pybullet.enableJointForceTorqueSensor(bodyUniqueId=self.id, jointIndex=self.RL_soleid, enableSensor=1)

        self.FR_sensor = []
        self.FL_sensor = []
        self.RR_sensor = []
        self.RL_sensor = []


    def read_data_offline(self, FilePathName):
        Input = open(FilePathName, 'r')
        maximan_column = 3
        datax = []
        for i in range(0, maximan_column):
            try:
                a = Input.readline()
                ax = a.split()
                datax.append(ax)
            except:
                pass
        # data = map(map,[float,float,float,float,float,float,float,float,float,float], data)
        data = np.array(datax)
        print('data size', data.shape)

        return data

    # def get_acutated_joint_pos_vel(self):
    #     acutated_joint_pos = []
    #     acutated_joint_vel = []
    #
    #     for i in range(0,self.jointactuatednumber):
    #         acutated_joint_pos.append(self.robot.getActuatedJointNames(self.id, self.jointactuatedindex[i])[0])
    #         acutated_joint_vel.append(self.robot.getActuatedJointNames(self.id, self.jointactuatedindex[i])[1])
    #
    #
    #     return acutated_joint_pos, acutated_joint_vel
            
            
            


    def state_estimation(self, i, dt, support_flag, links_pos_prev, links_vel_prev, gcom_pre):
        ### CoM position######
        gcom, FR_sole_pose, FL_sole_pose, RR_sole_pose, RL_sole_pose, base_pos, base_angle = self.cal_com_state()

        self.ankle_joint_pressure()

        links_pos, links_vel, links_acc = self.get_link_vel_vol(i, dt, links_pos_prev, links_vel_prev)
        # gcop, support_flag, dcm_pos, com_vel = self.cal_cop(i, support_flag, links_pos, links_acc, right_ankle_force,
        #                                                     left_ankle_force, right_sole_pos, left_sole_pos, gcom_pre,
        #                                                     gcom, dt)

        return gcom, FR_sole_pose, FL_sole_pose, RR_sole_pose, RL_sole_pose, base_pos, base_angle, self.FR_sensor, self.FL_sensor, self.RR_sensor, self.RL_sensor, links_pos, links_vel, links_acc

    ####### state feedback and recomputation#####################################
    ###### com state, foot location, baseposition and orientation===========================
    def cal_com_state(self):
        total_mass_moment = [0, 0, 0]
        com_pos = [0, 0, 0]

        for linkId in range(0, self.jointnumber):
            link_com_pos = pybullet.getLinkState(self.id, linkId)[0]
            total_mass_moment[0] += link_com_pos[0] * self.robot.getLinkMass(linkId)
            total_mass_moment[1] += link_com_pos[1] * self.robot.getLinkMass(linkId)
            total_mass_moment[2] += link_com_pos[2] * self.robot.getLinkMass(linkId)

        base_link_pos,base_orn = pybullet.getBasePositionAndOrientation(self.id)
        total_mass_moment[0] += base_link_pos[0] * self.robot.getLinkMass(-1)
        total_mass_moment[1] += base_link_pos[1] * self.robot.getLinkMass(-1)
        total_mass_moment[2] += base_link_pos[2] * self.robot.getLinkMass(-1)

        # total_mass_moment = [total_mass_momentx,total_mass_momenty,total_mass_momentz]
        com_pos[0] = total_mass_moment[0] / self.mass
        com_pos[1] = total_mass_moment[1] / self.mass
        com_pos[2] = total_mass_moment[2] / self.mass

        ###### footsole position: eliminating the offset
        ###### using the fact the linkid = jointid
        FR_sole_link_id = self.Link_name_index_dict.get('FR_foot_sole')
        FL_sole_link_id = self.Link_name_index_dict.get('FL_foot_sole')
        RR_sole_link_id = self.Link_name_index_dict.get('RR_foot_sole')
        RL_sole_link_id = self.Link_name_index_dict.get('RL_foot_sole')

        FR_sole_pose = list(pybullet.getLinkState(self.id, self.FR_soleid)[0])
        FL_sole_pose = list(pybullet.getLinkState(self.id, self.FL_soleid)[0])
        RR_sole_pose = list(pybullet.getLinkState(self.id, self.RR_soleid)[0])
        RL_sole_pose = list(pybullet.getLinkState(self.id, self.RL_soleid)[0])

        base_angle = pybullet.getEulerFromQuaternion(base_orn)

        return com_pos, FR_sole_pose, FL_sole_pose, RR_sole_pose, RL_sole_pose, base_link_pos, base_angle


    ####### ankle joint force/torque sensor
    def ankle_joint_pressure(self):
        self.FR_sensor = pybullet.getJointState(bodyUniqueId=self.id, jointIndex=self.FR_soleid)[2]
        self.FL_sensor = pybullet.getJointState(bodyUniqueId=self.id, jointIndex=self.FL_soleid)[2]
        self.RR_sensor = pybullet.getJointState(bodyUniqueId=self.id, jointIndex=self.RR_soleid)[2]
        self.RL_sensor = pybullet.getJointState(bodyUniqueId=self.id, jointIndex=self.RL_soleid)[2]



    ##### getlink velocity and acceleration
    def get_link_vel_vol(self, i, dt, links_pos_pre, links_vel_pre):
        links_pos = np.zeros([self.jointnumber + 1, 3])
        links_vel = np.zeros([self.jointnumber + 1, 3])
        links_acc = np.zeros([self.jointnumber + 1, 3])
        for linkId in range(0, self.jointnumber):
            links_pos[linkId] = pybullet.getLinkState(self.id, linkId, computeLinkVelocity=1)[0]
            links_vel[linkId] = pybullet.getLinkState(self.id, linkId, computeLinkVelocity=1)[6]  #### link velocity

        links_pos[self.jointnumber] = pybullet.getBasePositionAndOrientation(self.id)[0]
        links_vel[self.jointnumber] = pybullet.getBaseVelocity(self.id)[0]

        if (i >= 1):
            links_acc = (links_vel - links_vel_pre) / dt

        return links_pos, links_vel, links_acc

    ##### ZMP/DCM calculation
    def cal_cop(self, i, support_flag, links_pos, links_acc, right_ankle_force, left_ankle_force, right_sole_pos,
                left_sole_pos, gcom_pre, gcom, dt):
        total_vetical_acce = 0.0
        total_vetical_accx = 0.0
        total_vetical_accy = 0.0
        total_forwar_accz = 0.0
        total_lateral_accz = 0.0

        cop_state = [0, 0, 0]
        com_vel = [0, 0, 0]
        dcm_pos = [0, 0, 0]

        if (i == 0):

            support_flag[i] = 0
            dcm_pos[0] = gcom[0]
            dcm_pos[1] = gcom[1]
            com_vel = [0, 0, 0]
        else:
            #### computing the pz:
            if (abs(right_ankle_force[2]) > 100):  #### right leg is touching the ground
                if (abs(left_ankle_force[2]) > 100):  #### left leg is touching the ground
                    support_flag[i] = 0
                    if (support_flag[
                        i - 1] <= 1):  ###### from right support switching to left support, taking the right support as the current height
                        cop_state[2] = right_sole_pos[2]
                        flagxxx = 1
                    else:
                        cop_state[2] = left_sole_pos[
                            2]  ###### from left support switching to right support, taking the left support as the current height
                        flagxxx = 2

                else:  ###right support
                    cop_state[2] = right_sole_pos[2]
                    support_flag[i] = 1  ### right support
                    flagxxx = 3
            else:
                if (abs(left_ankle_force[2])) > 100:  #### left leg is touching the ground
                    cop_state[2] = left_sole_pos[2]
                    support_flag[i] = 2  ### left support
                    flagxxx = 4
                else:
                    cop_state[2] = (right_sole_pos[2] + left_sole_pos[2]) / 2
                    support_flag[i] = 0
                    flagxxx = 5

            for linkId in range(0, self.jointnumber + 1):
                total_vetical_acce += (links_acc[linkId, 2] + self.g)
                total_vetical_accx += ((links_acc[linkId, 2] + self.g) * links_pos[linkId, 0])
                total_vetical_accy += ((links_acc[linkId, 2] + self.g) * links_pos[linkId, 1])
                total_forwar_accz += ((links_pos[linkId, 2] - cop_state[2]) * links_acc[linkId, 0])
                total_lateral_accz += ((links_pos[linkId, 2] - cop_state[2]) * links_acc[linkId, 1])

            cop_state[0] = (total_vetical_accx - total_forwar_accz) / total_vetical_acce
            cop_state[1] = (total_vetical_accy - total_lateral_accz) / total_vetical_acce

            com_vel[0] = (gcom[0] - gcom_pre[0]) / dt
            com_vel[1] = (gcom[1] - gcom_pre[1]) / dt
            com_vel[2] = (gcom[2] - gcom_pre[2]) / dt

            dcm_omega = np.sqrt(self.g / (gcom[2] - cop_state[2]))
            dcm_pos[0] = gcom[0] + 1.0 / dcm_omega * com_vel[0]
            dcm_pos[1] = gcom[1] + 1.0 / dcm_omega * com_vel[1]
            dcm_pos[2] = cop_state[2]

        return cop_state, support_flag, dcm_pos, com_vel

    ##### controller-block#############################
    ###### IK-based controller
    ##### CoM/Body PD controller
    def CoM_Body_pd(self,dt,com_ref_det,com_feedback_det,com_ref_det_pre,com_feedback_det_pre,angle_ref_det,angle_feedback_det,angle_ref_det_pre,angle_feedback_det_pre):
        det_com = [0,0,0]
        det_body_angle = [0, 0, 0]
        det_com[0] = 0.2* (com_ref_det[0]-com_feedback_det[0]) +  0.00001* (com_ref_det[0]-com_feedback_det[0] - (com_ref_det_pre[0]-com_feedback_det_pre[0]))/dt
        det_com[1] = 0.05* (com_ref_det[1]-com_feedback_det[1]) +  0.00001* (com_ref_det[1]-com_feedback_det[1] - (com_ref_det_pre[1]-com_feedback_det_pre[1]))/dt
        det_com[2] = 0.1* (com_ref_det[2]-com_feedback_det[2]) +  0.00001* (com_ref_det[2]-com_feedback_det[2] - (com_ref_det_pre[2]-com_feedback_det_pre[2]))/dt
        det_body_angle[0] = 0.01* (angle_ref_det[0]-angle_feedback_det[0]) +  0.00001* (angle_ref_det[0]-angle_feedback_det[0] - (angle_ref_det_pre[0]-angle_feedback_det_pre[0]))/dt
        det_body_angle[1] = 0.1* (angle_ref_det[1]-angle_feedback_det[1]) +  0.00001* (angle_ref_det[1]-angle_feedback_det[1] - (angle_ref_det_pre[1]-angle_feedback_det_pre[1]))/dt
        det_body_angle[2] = 0.01* (angle_ref_det[2]-angle_feedback_det[2]) +  0.00001* (angle_ref_det[2]-angle_feedback_det[2] - (angle_ref_det_pre[2]-angle_feedback_det_pre[2]))/dt

        return det_com, det_body_angle

    ####### Rotation matrix generated by the  rpy angle
    def RotMatrixfromEuler(self, xyz):
        x_angle = xyz[0]
        y_angle = xyz[1]
        z_angle = xyz[2]
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

    def CoM_pos_fb_tracking(self,body_p_ref, body_v_ref,body_p_mea, body_v_mea,body_r_ref, body_rv_ref,body_r_mea, body_rv_mea):
        body_v_ref = np.zeros([3,1])
        body_rv_ref = np.zeros([3, 1])
        det_p = self.kpp * (body_p_ref - body_p_mea) + self.kdp * (body_v_ref - body_v_mea)
        det_p[0,0] = self.kppx * (body_p_ref[0,0] - body_p_mea[0,0]) + self.kdpx * (body_v_ref[0,0] - body_v_mea[0,0])
        det_r = self.kpr * (body_r_ref - body_r_mea) + self.kdr * (body_rv_ref - body_rv_mea)

        if(det_p[0,0]>0.005):
            det_p[0, 0] = 0.005
        elif(det_p[0,0]<-0.005):
            det_p[0, 0] = -0.005

        if(det_p[1,0]>0.005):
            det_p[1, 0] = 0.005
        elif(det_p[1,0]<-0.005):
            det_p[1, 0] = -0.005

        if(det_p[2,0]>0.005):
            det_p[2, 0] = 0.005
        elif(det_p[2,0]<-0.005):
            det_p[2, 0] = -0.005

        if(det_r[0,0]>0.1):
            det_r[0, 0] = 0.1
        elif(det_r[0,0]<-0.1):
            det_r[0, 0] = -0.1

        if(det_r[1,0]>0.1):
            det_r[1, 0] = 0.1
        elif(det_r[1,0]<-0.1):
            det_r[1, 0] = -0.1

        if(det_r[2,0]>0.1):
            det_r[2, 0] = 0.1
        elif(det_r[2,0]<-0.1):
            det_r[2, 0] = -0.1


        # print("det_p_pd:",det_p.T)

        return det_p, det_r
