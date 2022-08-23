#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# __author__ = "Songyan Xin"
# __copyright__ = "Copyright (C) 2020 Songyan Xin"

import numpy as np
import pybullet
import pybullet_data




class Color:
    red = [1,0,0]
    green = [0,1,0]
    blue = [0,0,1]


class ControlMode:
    positionControl = 'positionControl'
    torqueControl = 'torqueControl'



class SimRobot:
    def __init__(self, urdfFileName, basePosition=[0,0,0], baseRPY=[0,0,0], jointPositions=None, useFixedBase=False, verbose=True, Torquecontrol=True):

        self.id = pybullet.loadURDF(fileName=urdfFileName,
                                    basePosition=basePosition,
                                    baseOrientation=pybullet.getQuaternionFromEuler(baseRPY),
                                    useFixedBase=useFixedBase,
                                    flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

        if verbose:
            print('*' * 100 + '\nPyBullet Robot Info ' + '\u2193 '*20 + '\n' + '*' * 100)
            print('robot ID:              ', self.id)
            print('robot name:            ', self.getRobotName())
            print('robot total mass:      ', self.getRobotMass())
            print('base link name:        ', self.getBaseName())
            print('num of joints:         ', self.getNumJoints())
            print('num of actuated joints:', self.getNumActuatedJoints())
            print('joint names:           ', len(self.getJointNames()), self.getJointNames())
            print('joint indexes:         ', len(self.getJointIndexes()), self.getJointIndexes())
            print('actuated joint names:  ', len(self.getActuatedJointNames()), self.getActuatedJointNames())
            print('actuated joint indexes:', len(self.getActuatedJointIndexes()), self.getActuatedJointIndexes())
            print('link names:            ', len(self.getLinkNames()), self.getLinkNames())
            print('link indexes:          ', len(self.getLinkIndexes()), self.getLinkIndexes())
            print('joint dampings:        ', self.getJointDampings())
            print('joint frictions:       ', self.getJointFrictions())
            print('*' * 100 + '\nPyBullet Robot Info ' + '\u2191 '*20 + '\n' + '*' * 100)

        if(Torquecontrol):
            self.enableTorqueControl()
        else:
            self.enablePositionControl()

        # self.addDebugLinkFrames()

        if jointPositions is None:
            self.resetJointStates(np.zeros(self.getNumActuatedJoints()))
            # self.resetJointStates(np.zeros(self.getNumJoints()))
        else:
            self.resetJointStates(jointPositions)





    def setfixedbase(self,kinematic_test=False):
        basePosition = self.getBaseCoMPosition()
        # basePosition = self.getBaseLinkPosition()
        if kinematic_test:
            pybullet.createConstraint(self.id, -1, -1, -1, pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], basePosition)
        else:
            pass


    def resetBasePose(self, position=[0,0,0], quaternion=[0,0,0,1]):
        pybullet.resetBasePositionAndOrientation(self.id, position, quaternion)

    def resetBaseVelocity(self, linear=[0,0,0], angular=[0,0,0]):
        pybullet.resetBaseVelocity(self.id, linear, angular)

    def resetBaseStates(self, position=[0,0,0], quaternion=[0,0,0,1], velocityLinear=[0,0,0], velocityAngular=[0,0,0]):
        pybullet.resetBasePositionAndOrientation(self.id, position, quaternion)
        pybullet.resetBaseVelocity(self.id, velocityLinear, velocityAngular)

    def resetJointStates(self, jointPositions, jointVelocities=None):
        if jointVelocities is None:
            jointVelocities = np.zeros(self.getNumActuatedJoints())
        for jointIndex, jointPosition, jointVelocity in zip(self.getActuatedJointIndexes(), jointPositions, jointVelocities):
            pybullet.resetJointState(self.id, jointIndex, jointPosition, jointVelocity)

    def resetJointDampings(self, jointDamping=0.0):
        for i in range(self.getNumJoints()):
            pybullet.changeDynamics(bodyUniqueId=self.id, linkIndex=i, jointDamping=jointDamping)

    def enablePositionControl(self):
        pybullet.setJointMotorControlArray(self.id,
                                           self.getJointIndexes(),
                                           pybullet.POSITION_CONTROL,
                                           targetPositions=[0.0] * self.getNumJoints())
        self.controlMode = ControlMode.positionControl
        print(self.controlMode, 'enabled!')

    def enableTorqueControl(self):
        pybullet.setJointMotorControlArray(self.id,
                                           self.getJointIndexes(),
                                           pybullet.VELOCITY_CONTROL,
                                           forces=[0.0]*self.getNumJoints())
        self.controlMode = ControlMode.torqueControl
        print(self.controlMode, 'enabled!')

    def getRobotMass(self):
        totalMass = 0
        for linkId in range(-1, self.getNumJoints()):
            totalMass += self.getLinkMass(linkId)
        return totalMass


    def getBaseName(self):
        return pybullet.getBodyInfo(self.id)[0].decode()

    def getRobotName(self):
        return pybullet.getBodyInfo(self.id)[1].decode()

    def getNumJoints(self):
        return pybullet.getNumJoints(self.id)

    def getJointIndex(self, jointName):
        return self.getJointNameIndexMap()[jointName]

    def getJointIndexes(self):
        return list(range(self.getNumJoints()))

    def getJointName(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[1].decode()

    def getJointNames(self):
        try:
            return self.joint_names
        except AttributeError:
            self.joint_names = []
            for i in range(self.getNumJoints()):
                joint_name = self.getJointName(i)
                self.joint_names.append(joint_name)
            return self.joint_names

    def getLinkIndex(self, linkName):
        return self.getLinkNameIndexMap()[linkName]

    def getLinkIndexes(self):
        return list(range(self.getNumJoints()))

    def getLinkName(self, linkIndex):
        return pybullet.getJointInfo(self.id, linkIndex)[12].decode()


    def getLinkNames(self):
        try:
            return self.linkNames
        except AttributeError:
            self.linkNames = []
            for i in range(self.getNumJoints()):
                linkName = self.getLinkName(i)
                self.linkNames.append(linkName)
            return self.linkNames

    def getJointType(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[2]

    def getJointDamping(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[6]

    def getJointDampings(self):
        joint_dampings = []
        for i in range(self.getNumJoints()):
            joint_dampings.append(self.getJointDamping(i))
        return joint_dampings

    def getJointFriction(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[7]

    def getJointFrictions(self):
        joint_frictions = []
        for i in range(self.getNumJoints()):
            joint_frictions.append(self.getJointFriction(i))
        return joint_frictions

    def getJointLowerLimit(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[8]

    def getJointLowerLimits(self):
        jointLowerLimits = []
        for i in self.getJointIndexes():
            jointLowerLimits.append(self.getJointLowerLimit(i))
        return jointLowerLimits

    def getJointUpperLimit(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[9]

    def getJointUpperLimits(self):
        jointUpperLimits = []
        for i in self.getJointIndexes():
            jointUpperLimits.append(self.getJointUpperLimit(i))
        return jointUpperLimits

    def getJointMaxForce(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[10]

    def getJointMaxVelocity(self, jointIndex):
        return pybullet.getJointInfo(self.id, jointIndex)[11]


    def getJointIndexNameMap(self):
        try:
            return self.joint_index_name_map
        except AttributeError:
            joint_indexes = self.getJointIndexes()
            joint_names = self.getJointNames()
            self.joint_index_name_map = dict(zip(joint_indexes, joint_names))
            return self.joint_index_name_map

    def getLinkIndexNameMap(self):
        try:
            return self.linkIndexNameMap
        except AttributeError:
            linkIndexes = self.getLinkIndexes()
            linkNames = self.getLinkNames()
            self.linkIndexNameMap = dict(zip(linkIndexes, linkNames))
            return self.linkIndexNameMap

    def getLinkNameIndexMap(self):
        try:
            return self.linkNameIndexMap
        except AttributeError:
            linkIndexes = self.getLinkIndexes()
            linkNames = self.getLinkNames()
            self.linkNameIndexMap = dict(zip(linkNames, linkIndexes))
            return self.linkNameIndexMap

    def getJointNameIndexMap(self):
        try:
            return self.joint_name_index_map
        except AttributeError:
            joint_indexes = self.getJointIndexes()
            joint_names = self.getJointNames()
            self.joint_name_index_map = dict(zip(joint_names, joint_indexes))
        return self.joint_name_index_map


    def getNumActuatedJoints(self):
        n = 0
        for i in range(self.getNumJoints()):
            if self.getJointType(i) is not pybullet.JOINT_FIXED:
                n += 1
        return n

    def getActuatedJointNames(self):
        try:
            return self.actuated_joint_names
        except AttributeError:
            self.actuated_joint_names = []
            for i in range(self.getNumJoints()):
                if self.getJointType(i) is not pybullet.JOINT_FIXED:
                    self.actuated_joint_names.append(self.getJointName(i))
            return self.actuated_joint_names

    def getActuatedJointIndexes(self):
        try:
            return self.actuated_joint_indexes
        except AttributeError:
            self.actuated_joint_indexes = []
            for joint_name in self.getActuatedJointNames():
                self.actuated_joint_indexes.append(self.getJointIndex(joint_name))
            return self.actuated_joint_indexes

    def getLinkMass(self, link_id):
        return pybullet.getDynamicsInfo(self.id, link_id)[0]

    def getLinkLocalInertialTransform(self, link_id):
        return pybullet.getDynamicsInfo(self.id, link_id)[3:5]

    def getLinkLocalInertialPosition(self, link_id):
        return np.array(pybullet.getDynamicsInfo(self.id, link_id)[3])

    def getLinkLocalInertiaQuaternion(self, link_id):
        return np.array(pybullet.getDynamicsInfo(self.id, link_id)[4])

    # Get Base states

    def getBaseLocalInertiaTransform(self):
        return self.getLinkLocalInertialTransform(-1)

    def getBaseLocalInertiaPosition(self):
        return self.getLinkLocalInertialPosition(-1)

    def getBaseLocalInertiaQuaternion(self):
        return self.getLinkLocalInertiaQuaternion(-1)

    def getBaseCoMPosition(self):
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[0])

    def getBaseCoMQuaternion(self):
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[1])

    def getBaseCoMTransform(self):
        return pybullet.getBasePositionAndOrientation(self.id)

    def getBaseLinkPosition(self):
        worldTransCom = self.getBaseCoMTransform()
        localTransCom = self.getBaseLocalInertiaTransform()
        comTransLocal = pybullet.invertTransform(position=localTransCom[0], orientation=localTransCom[1])
        worldTransLocal = pybullet.multiplyTransforms(positionA=worldTransCom[0], orientationA=worldTransCom[1],
                                                      positionB=comTransLocal[0], orientationB=comTransLocal[1])
        return worldTransLocal[0]

    def getBaseLinkQuaternion(self):
        return np.array(pybullet.getBasePositionAndOrientation(self.id)[1])

    def getBaseVelocityLinear(self):
        return pybullet.getBaseVelocity(self.id)[0]

    def getBaseVelocityAngular(self):
        return pybullet.getBaseVelocity(self.id)[1]

    def getBaseStates(self):
        return {'position':self.getBaseLinkPosition(),
                'quaternion':self.getBaseLinkQuaternion(),
                'velocityLinear':self.getBaseVelocityLinear(),
                'velocityAngular':self.getBaseVelocityAngular()}

    # Get joint states

    def getJointPositions(self):
        jointPositions = np.array([state[0] for state in pybullet.getJointStates(self.id, self.getJointIndexes())])
        return dict(zip(self.getJointNames(), jointPositions))

    def getJointVelocities(self):
        jointVelocities = np.array([state[1] for state in pybullet.getJointStates(self.id, self.getJointIndexes())])
        return dict(zip(self.getJointNames(), jointVelocities))

    def getJointStates(self):
        jointStates = [state[:2] for state in pybullet.getJointStates(self.id, self.getJointIndexes())]
        return dict(zip(self.getJointNames(), jointStates))

    def getActuatedJointPositions(self):
        actuatedJointPositions = np.array([state[0] for state in pybullet.getJointStates(self.id, self.getActuatedJointIndexes())])
        # return dict(zip(self.getActuatedJointNames(), actuatedJointPositions))
        return actuatedJointPositions

    def getActuatedJointVelocities(self):
        actuatedJointVelocities = np.array([state[1] for state in pybullet.getJointStates(self.id, self.getActuatedJointIndexes())])
        #return dict(zip(self.getActuatedJointNames(), actuatedJointVelocities))
        return actuatedJointVelocities

    def getActuatedJointtorques(self):
        actuatedJointtorques = np.array([state[3] for state in pybullet.getJointStates(self.id, self.getActuatedJointIndexes())])
        # return dict(zip(self.getActuatedJointNames(), actuatedJointtorques))
        return actuatedJointtorques

    def getActuatedJointStates(self):
        actuatedJointStates = [state[:2] for state in pybullet.getJointStates(self.id, self.getActuatedJointIndexes())]
        return dict(zip(self.getActuatedJointNames(), actuatedJointStates))

    def getRobotStates(self):
        return {'baseStates':self.getBaseStates(),
                'actuatedJointPositions':self.getActuatedJointPositions(),
                'actuatedJointVelocities':self.getActuatedJointVelocities()}

    # Set control commands
    def setActuatedJointPositions(self, jointPositions):
        if self.controlMode is not ControlMode.positionControl:
            self.enablePositionControl()
        if isinstance(jointPositions, dict):
            actuatedJointPositions = [jointPositions[jointName] for jointName in self.getActuatedJointNames()]
            pybullet.setJointMotorControlArray(self.id, self.getActuatedJointIndexes(), pybullet.POSITION_CONTROL, targetPositions=actuatedJointPositions)
        elif isinstance(jointPositions, np.ndarray) or isinstance(jointPositions, list) or isinstance(jointPositions, tuple):
            pybullet.setJointMotorControlArray(self.id, self.getActuatedJointIndexes(), pybullet.POSITION_CONTROL, targetPositions=jointPositions)

    def setActuatedJointTorques(self, jointTorques):
        if self.controlMode is not ControlMode.torqueControl:
            self.enableTorqueControl()
        if isinstance(jointTorques, dict):
            actuatedJointTorques = [jointTorques[jointName] for jointName in self.getActuatedJointNames()]
            pybullet.setJointMotorControlArray(self.id, self.getActuatedJointIndexes(), pybullet.TORQUE_CONTROL, forces=actuatedJointTorques)
        elif isinstance(jointTorques, np.ndarray) or isinstance(jointTorques, list) or isinstance(jointTorques, tuple):
            pybullet.setJointMotorControlArray(self.id, self.getActuatedJointIndexes(), pybullet.TORQUE_CONTROL, forces=jointTorques)


    # Debug

    def addDebugLinkFrames(self, axisLength=0.1, axisWidth=1):
        for linkId in range(-1, self.getNumJoints()):
            self.addDebugLinkFrame(linkId, axisLength, axisWidth)

    def addDebugLinkInertiaFrames(self, axisLength=0.05, axisWidth=5):
        for linkId in range(-1, self.getNumJoints()):
            self.addDebugLinkInertiaFrame(linkId, axisLength=axisLength, axisWidth=axisWidth)

    def addDebugLinkFrame(self, linkId, axisLength=0.2, axisWidth=1):
        localTransCom = self.getLinkLocalInertialTransform(linkId)
        comPosLocal, comQuatLocal = pybullet.invertTransform(localTransCom[0], localTransCom[1])
        comRotLocal = np.array(pybullet.getMatrixFromQuaternion(comQuatLocal)).reshape((3,3))
        pybullet.addUserDebugLine(comPosLocal, comPosLocal+comRotLocal[:,0] * axisLength, Color.red, lineWidth=axisWidth, parentObjectUniqueId=self.id, parentLinkIndex=linkId)
        pybullet.addUserDebugLine(comPosLocal, comPosLocal+comRotLocal[:,1] * axisLength, Color.green, lineWidth=axisWidth, parentObjectUniqueId=self.id, parentLinkIndex=linkId)
        pybullet.addUserDebugLine(comPosLocal, comPosLocal+comRotLocal[:,2] * axisLength, Color.blue, lineWidth=axisWidth, parentObjectUniqueId=self.id, parentLinkIndex=linkId)


    def addDebugLinkInertiaFrame(self, linkId, axisLength=0.2, axisWidth=1):
        position, quaternion = [0, 0, 0], [0, 0, 0, 1]
        rotation = np.array(pybullet.getMatrixFromQuaternion(quaternion)).reshape((3, 3))
        pybullet.addUserDebugLine(position, position+rotation[:,0] * axisLength, Color.red, lineWidth=axisWidth, parentObjectUniqueId=self.id, parentLinkIndex=linkId)
        pybullet.addUserDebugLine(position, position+rotation[:,1] * axisLength, Color.green, lineWidth=axisWidth, parentObjectUniqueId=self.id, parentLinkIndex=linkId)
        pybullet.addUserDebugLine(position, position+rotation[:,2] * axisLength, Color.blue, lineWidth=axisWidth, parentObjectUniqueId=self.id, parentLinkIndex=linkId)

    def addDebugFrame(self, position, quaternion, axisLength=0.2, axisWidth=1):
        position, quaternion = [0, 0, 0], [0, 0, 0, 1]
        rotation = np.array(pybullet.getMatrixFromQuaternion(quaternion)).reshape((3, 3))
        pybullet.addUserDebugLine(position, position + rotation[:, 0] * axisLength, Color.red, lineWidth=axisWidth)
        pybullet.addUserDebugLine(position, position + rotation[:, 1] * axisLength, Color.green, lineWidth=axisWidth)
        pybullet.addUserDebugLine(position, position + rotation[:, 2] * axisLength, Color.blue, lineWidth=axisWidth)


    def calculateInverseKinematics(self, linkName, position, rpy=None):

        if rpy is None:
            jointPositions = pybullet.calculateInverseKinematics(self.id, self.getLinkIndex(linkName), position)
        else:
            quaternion = pybullet.getQuaternionFromEuler(rpy)
            lowerLimits = np.array(self.getJointLowerLimits())
            upperLimits = np.array(self.getJointUpperLimits())
            jointRanges = upperLimits - lowerLimits
            restPoses = (lowerLimits + upperLimits)/2.0

            # jointPositions = pybullet.calculateInverseKinematics(self.id, self.getLinkIndex(linkName), position, quaternion, lowerLimits, upperLimits, jointRanges, restPoses)
            jointPositions = pybullet.calculateInverseKinematics(self.id, self.getLinkIndex(linkName), position, quaternion)

        return jointPositions

if __name__ == '__main__':

    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=500, nanstr='nan', precision=5, suppress=False, threshold=1000, formatter=None)

    from sim_env import SimEnv

    sim_env = SimEnv()

    dataFolder = pybullet_data.getDataPath()
    urdf_filename = dataFolder + '/kuka_iiwa/model.urdf'

    robot = SimRobot(urdfFileName=urdf_filename, basePosition=[0,0,0], useFixedBase=True)
    robot.enablePositionControl()
    # robot.enableTorqueControl()

    jointPositions = robot.calculateInverseKinematics('lbr_iiwa_link_7', [0.7, 0.0, 0.3], [0, np.pi/2, 0])
    robot.setActuatedJointPositions(jointPositions)


    while True:
        # print(robot.getJointPositions())
        # print(robot.getJointVelocities())
        # print(robot.getActuatedJointVelocities())
        # print(robot.getJointStates())
        # print(robot.getActuatedJointStates())
        # print(robot.getRobotStates())


        sim_env.step()
        sim_env.debug()