from .base_kinematics import BaseKinematics
from .utils.bennett_ik import GeneralBennettIK
import numpy as np
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import math
from math import sqrt, atan, asin, acos, cos

class MagicLimbPlanarKin(BaseKinematics, GeneralBennettIK):
    def __init__(self):
        urdf_path = "MagicLimbRobotPlanarFL.urdf"
        tld = os.path.join(CURRENT_DIR, 'urdfs', 'MagicLimbPlanar')

        link_len = 0.16252 # Theoretical length of the links
        foot_x0 = 0.37643424 # x position of foot in initial configuration
        foot_y0 = 0.02215343 # y position of foot in initial configuration
        foot_z0 = 0.07 # z position of foot in initial configuration

        screw_hip = np.array([1, 0, 0, 0, 0, 0])
        screw_thigh = np.array([0, 0, 1, 0, 0, 0])
        screw_knee = np.array([0, 0, 1, 0, -link_len, 0])

        M0 = np.eye(4)
        M0[0, 3] = foot_x0
        M0[1, 3] = foot_y0
        M0[2, 3] = foot_z0

        self.l = link_len
        self.x0 = foot_x0
        self.y0 = foot_y0
        self.z0 = foot_z0

        self.__thigh_offset = (9.17191/2.0) * np.pi / 180.0  # Motor thigh angular offset in radians
        self.__knee_offset = 9.17191 * np.pi / 180.0  # Motor knee angular offset in radians

        BaseKinematics.__init__(
            self,
            span_tree_urdf_path=urdf_path, 
            span_tree_urdf_tld=tld,
            screw_hip=screw_hip, 
            screw_thigh=screw_thigh, 
            screw_knee=screw_knee, 
            M0=M0,
        )

        GeneralBennettIK.__init__(
            self,
            l=self.l,
            x0=self.x0,
            y0=self.y0,
            z0=self.z0,
            zt=self.z0,  # z position of the knee joint origin
            beta=0.0,  # Bennett linkage beta
            thigh_offset=self.__thigh_offset,
            knee_offset=self.__knee_offset
        )

    # Implement mapping functions between actuator and span tree joint positions
    def span_tree_joint_pos_from_actuator(self, actuator_joint_pos):
        span_tree_joint_pos = np.zeros(3)
        # Hip joint position remains the same
        span_tree_joint_pos[0] = actuator_joint_pos[0]

        # For thigh joint, add the offset
        span_tree_joint_pos[1] = actuator_joint_pos[1] + self.__thigh_offset

        # For knee joint, subtract the offset
        span_tree_joint_pos[2] = actuator_joint_pos[2] - self.__knee_offset

        return span_tree_joint_pos
    
    def actuator_joint_pos_from_span_tree(self, span_tree_joint_pos):
        actuator_joint_pos = np.zeros(3)
        # Hip joint position remains the same
        actuator_joint_pos[0] = span_tree_joint_pos[0]

        # For thigh joint, subtract the offset
        actuator_joint_pos[1] = span_tree_joint_pos[1] - self.__thigh_offset

        # For calf joint, add the offset
        actuator_joint_pos[2] = span_tree_joint_pos[2] + self.__knee_offset

        return actuator_joint_pos
    
    def actuator_joint_vel_from_span_tree(self, span_tree_joint_pos, span_tree_joint_vel):
        '''No mapping is needed'''
        return span_tree_joint_vel.copy()
    
    def span_tree_joint_vel_from_actuator(self, actuator_joint_pos, actuator_joint_vel):
        return actuator_joint_vel.copy()
    
    def analytic_ik(self, goal_pos_in_hip_frame):
        span_tree_solution, success, ik_solutions, ik_errors = self.analytical_inverse_kinematics_span_tree(goal_pos_in_hip_frame)
        actuator_solution = self.actuator_joint_pos_from_span_tree(span_tree_solution)
        return actuator_solution, success, ik_solutions, ik_errors