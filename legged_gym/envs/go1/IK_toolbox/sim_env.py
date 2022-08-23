#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# __author__ = "Songyan Xin"
# __copyright__ = "Copyright (C) 2020 Songyan Xin"

import time
import pybullet
import pybullet_data

class Color:
    black = [0,0,0]
    red = [1,0,0]
    green = [0,1,0]
    blue = [0,0,1]
    white = [1,1,1]

    @classmethod
    def rand(self):
        import random
        return [random.random()]*3


class Floor:
    def __init__(self, basePosition=[0,0,0], baseRPY=[0,0,0]):
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.id = pybullet.loadURDF('plane.urdf', basePosition=basePosition, baseOrientation=pybullet.getQuaternionFromEuler(baseRPY))


    def changeFriction(self, lateralFriction=1.0, spinningFriction=1.0):
        pybullet.changeDynamics(bodyUniqueId=self.id, linkIndex=-1, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
        print("Floor friction updated!")
        print("lateralFriction:", pybullet.getDynamicsInfo(self.id, -1)[1])
        print("spinningFriction:", pybullet.getDynamicsInfo(self.id, -1)[7])


class Slope:
    def __init__(self, urdf, basePosition=[0,0,0], baseRPY=[0,0,0]):
        self.id = pybullet.loadURDF(urdf, basePosition=basePosition, baseOrientation=pybullet.getQuaternionFromEuler(baseRPY))

    def changeFriction(self, lateralFriction=1.0, spinningFriction=1.0):
        pybullet.changeDynamics(bodyUniqueId=self.id, linkIndex=-1, lateralFriction=lateralFriction, spinningFriction=spinningFriction)
        print("Floor friction updated!")
        print("lateralFriction:", pybullet.getDynamicsInfo(self.id, -1)[1])
        print("spinningFriction:", pybullet.getDynamicsInfo(self.id, -1)[7])


class Cube:
    def __init__(self, basePosition, baseRPY, useFixedBase=False, globalScaling=1.0):
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        table_urdf_path = pybullet_data.getDataPath() + "/cube_small.urdf"
        self.id = pybullet.loadURDF(table_urdf_path,
                                    basePosition=basePosition,
                                    baseOrientation=pybullet.getQuaternionFromEuler(baseRPY),
                                    useMaximalCoordinates=0,
                                    useFixedBase=useFixedBase,
                                    flags=0,
                                    globalScaling=globalScaling,
                                    physicsClientId=0)

class Table:
    def __init__(self, basePosition, baseRPY, useFixedBase, globalScaling=1.0):
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        table_urdf_path = pybullet_data.getDataPath() + "/table/table.urdf"
        self.id = pybullet.loadURDF(table_urdf_path,
                                    basePosition=basePosition,
                                    baseOrientation=pybullet.getQuaternionFromEuler(baseRPY),
                                    useMaximalCoordinates=0,
                                    useFixedBase=useFixedBase,
                                    flags=0,
                                    globalScaling=globalScaling,
                                    physicsClientId=0)


    def changeFriction(self, lateralFriction=1.0, spinningFriction=1.0, rollingFriction=0.0):
        print("Current table dynamic: ", pybullet.getDynamicsInfo(self.id, -1))
        pybullet.changeDynamics(bodyUniqueId=self.id, linkIndex=-1, lateralFriction=lateralFriction, spinningFriction=spinningFriction, rollingFriction=rollingFriction)
        print("Updated table dynamic: ", pybullet.getDynamicsInfo(self.id, -1))

class SimEnv:
    def __init__(self, sim_rate=200, g=9.8, real_time_sim=True):
        self.sim_rate = sim_rate
        self.sim_time_step = 1.0 / self.sim_rate
        self.real_time_sim = real_time_sim
        self.sim_count = 0
        self.sim_time = 0.0

        self.physics_client = pybullet.connect(pybullet.GUI)
        pybullet.setTimeStep(self.sim_time_step)
        pybullet.setGravity(0, 0, -g)

        self.floor = Floor()
        self.floor.changeFriction(lateralFriction=1.0, spinningFriction=1.0)


        self.configureDebugVisualizer(COV_ENABLE_GUI=False,
                                      COV_ENABLE_RGB_BUFFER_PREVIEW=False,
                                      COV_ENABLE_DEPTH_BUFFER_PREVIEW=False,
                                      COV_ENABLE_SEGMENTATION_MARK_PREVIEW=False)



    def configureDebugVisualizer(self, COV_ENABLE_GUI=False, COV_ENABLE_RGB_BUFFER_PREVIEW=False, COV_ENABLE_DEPTH_BUFFER_PREVIEW=False, COV_ENABLE_SEGMENTATION_MARK_PREVIEW=False):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, COV_ENABLE_GUI)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, COV_ENABLE_RGB_BUFFER_PREVIEW)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, COV_ENABLE_DEPTH_BUFFER_PREVIEW)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, COV_ENABLE_SEGMENTATION_MARK_PREVIEW)

    def reset(self):
        pass

    def resetCamera(self, cameraDistance=2.5, cameraYaw=0, cameraPitch=-10, cameraTargetPosition=[0,0,0.5]):
        pybullet.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                                            cameraYaw=cameraYaw,
                                            cameraPitch=cameraPitch,
                                            cameraTargetPosition=cameraTargetPosition)

    def debug(self):

        # pause simulation by space key event
        keys = pybullet.getKeyboardEvents()
        space_key = ord(' ')
        if space_key in keys and keys[space_key] & pybullet.KEY_WAS_TRIGGERED:
            print("*" * 100 + "Simulation Paused! Press 'Space' to resume!" + "*" * 100)
            while True:
                keys = pybullet.getKeyboardEvents()
                if space_key in keys and keys[space_key] & pybullet.KEY_WAS_TRIGGERED:
                    break


    def step(self):
        pybullet.stepSimulation()
        if self.real_time_sim:
            time.sleep(self.sim_time_step)
        self.sim_count += 1
        self.sim_time = self.sim_count * self.sim_time_step

    @staticmethod
    def addDebugLine(startPoint, endPoint, color=[0.5,0,0], lineWidth=1, lifeTime=0):
        return pybullet.addUserDebugLine(startPoint, endPoint, lineColorRGB=color, lineWidth=lineWidth, lifeTime=lifeTime)

    @staticmethod
    def addDebugRectangle(position, quaternion=[0,0,0,1], length=0.2, width=0.1, color=[0.5,0,0], lineWidth=1, lifeTime=0):
        point1, quaternion1 = pybullet.multiplyTransforms(position, quaternion, [+length / 2, +width / 2, 0], [0, 0, 0, 1])
        point2, quaternion2 = pybullet.multiplyTransforms(position, quaternion, [-length / 2, +width / 2, 0], [0, 0, 0, 1])
        point3, quaternion3 = pybullet.multiplyTransforms(position, quaternion, [-length / 2, -width / 2, 0], [0, 0, 0, 1])
        point4, quaternion4 = pybullet.multiplyTransforms(position, quaternion, [+length / 2, -width / 2, 0], [0, 0, 0, 1])
        line1 = SimEnv.addDebugLine(point1, point2, color, lineWidth, lifeTime)
        line2 = SimEnv.addDebugLine(point2, point3, color, lineWidth, lifeTime)
        line3 = SimEnv.addDebugLine(point3, point4, color, lineWidth, lifeTime)
        line4 = SimEnv.addDebugLine(point4, point1, color, lineWidth, lifeTime)
        return [line1, line2, line3, line4]


    @staticmethod
    def addDebugTrajectory(X, Y, Z, color=[0,0,0], lineWidth=1, lifeTime=0):
        trajectoryId = []
        for i in range(len(X)-1):
            pointFrom = [X[i], Y[i], Z[i]]
            pointTo = [X[i+1], Y[i+1], Z[i+1]]
            lineId = pybullet.addUserDebugLine(pointFrom, pointTo, lineColorRGB=color, lineWidth=lineWidth, lifeTime=lifeTime)
            trajectoryId.append(lineId)
        return trajectoryId

    @staticmethod
    def removeDebugItems(*args):
        '''
        remove one or multiple debug items
        :param args: int, list, tuple
            id of the items to be removed
        '''
        if len(args) == 0:
            pybullet.removeAllUserDebugItems()
        else:
            for arg in args:
                if isinstance(arg, int):
                    pybullet.removeUserDebugItem(arg)
                elif isinstance(arg, list) or isinstance(arg, tuple):
                    for item in arg:
                        SimEnv.removeDebugItems(item)

if __name__ == '__main__':
    sim_env = SimEnv()

    while True:

        sim_env.step()
        sim_env.debug()