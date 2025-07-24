from .base_kinematics import BaseKinematics
import numpy as np
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

import math
from math import sqrt, atan, asin, acos, cos

class MagicLimbPlanarKin(BaseKinematics):
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

        super().__init__(
            span_tree_urdf_path=urdf_path, 
            span_tree_urdf_tld=tld,
            screw_hip=screw_hip, 
            screw_thigh=screw_thigh, 
            screw_knee=screw_knee, 
            M0=M0,
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
    
    def analytic_ik_span_tree(self, goal_pos_in_hip_frame):
        '''Analytic inverse kinematics for the planar robot'''
        x, y, z = goal_pos_in_hip_frame
        
        A = sqrt(y**2 + z**2 - self.z0**2)
        B = sqrt((self.x0 - self.l)**2 + self.y0**2)
        PHI = atan(self.y0 / (self.x0 - self.l))
        E_sin_value = (x**2 + (self.x0-self.l)**2 + self.y0**2 + A**2 - self.l**2)/(2*B*sqrt(x**2 + A**2))
        assert abs(E_sin_value) <= 1 + 1e-3, "Inverse kinematics solution is not feasible"
        E_sin_value = np.clip(E_sin_value, -1, 1)  # Clip to avoid numerical issues
        E = asin(E_sin_value)
        F = atan(x/A)

        GAMMA1 = E + F
        GAMMA2 = E - F

        # q1_1 = -asin(z/sqrt(A**2 + self.z0**2)) - atan(self.z0 / A)
        # q1_2 = asin(z/sqrt(A**2 + self.z0**2)) - atan(self.z0 / A)
        # q1_3 = acos(y/sqrt(A**2 + self.z0**2)) - atan(-self.z0 / A)
        # q1_4 = acos(y/-sqrt(A**2 + self.z0**2)) - atan(-self.z0 / A)
        if (A*y + self.z0*z) == 0:
            q1_1 = -np.pi/2
        else:
            q1_1 = atan((A*z - self.z0*y)/(A*y + self.z0*z))
            if q1_1 > 0:
                q1_1 -= np.pi
        
        if -A*y + self.z0*z == 0:
            q1_2 = -np.pi/2
        else:
            q1_2 = atan((-A*z - self.z0*y)/(-A*y + self.z0*z))
            if q1_2 > 0:
                q1_2 -= np.pi

        # There are eight possible solutions, we choose the one with lowest error
        ik_solutions = np.zeros((16, 3))
        ik_solutions[0:8, 0] = q1_1
        ik_solutions[8:16, 0] = q1_2

        cos_value = (x - B * cos(GAMMA1))/self.l
        if abs(cos_value) <= 1:
            q2_1 = acos(cos_value)
            q3_1 = E + F - PHI - q2_1
            q3_2 = -E - F - PHI - q2_1
            ik_solutions[0, 1] = q2_1
            ik_solutions[0, 2] = q3_1
            ik_solutions[1, 1] = q2_1
            ik_solutions[1, 2] = q3_2

        cos_value = (x - B * cos(GAMMA2))/self.l
        if abs(cos_value) <= 1:
            q2_2 = acos(cos_value)
            q3_3 = E - F - PHI - q2_2
            q3_4 = -E + F - PHI - q2_2
            ik_solutions[2, 1] = q2_2
            ik_solutions[2, 2] = q3_3
            ik_solutions[3, 1] = q2_2
            ik_solutions[3, 2] = q3_4

        cos_value = (x + B * cos(GAMMA1))/self.l
        if abs(cos_value) <= 1:
            q2_3 = acos(cos_value)
            q3_5 = E + F - PHI - q2_3
            q3_6 = -E - F - PHI - q2_3
            ik_solutions[4, 1] = q2_3
            ik_solutions[4, 2] = q3_5
            ik_solutions[5, 1] = q2_3
            ik_solutions[5, 2] = q3_6

        cos_value = (x + B * cos(GAMMA2))/self.l
        if abs(cos_value) <= 1:
            q2_4 = acos(cos_value)
            q3_7 = E - F - PHI - q2_4
            q3_8 = -E + F - PHI - q2_4
            ik_solutions[6, 1] = q2_4
            ik_solutions[6, 2] = q3_7
            ik_solutions[7, 1] = q2_4
            ik_solutions[7, 2] = q3_8

        ik_solutions[8:16, 1:] = ik_solutions[0:8, 1:]

        ik_errors = np.zeros(ik_solutions.shape[0])
        # Check the ik error using forward kinematics
        for i in range(ik_solutions.shape[0]):
            pos, _ = self.forward_kinematics_span_tree(ik_solutions[i])
            ik_errors[i] = np.linalg.norm(pos - goal_pos_in_hip_frame)

        # Find the solution with the lowest error
        best_solution_index = np.argmin(ik_errors)
        best_solution = ik_solutions[best_solution_index]

        return best_solution, ik_errors[best_solution_index]
    
    def analytic_ik(self, goal_pos_in_hip_frame):
        span_tree_solution, error = self.analytic_ik_span_tree(goal_pos_in_hip_frame)
        actuator_solution = self.actuator_joint_pos_from_span_tree(span_tree_solution)
        return actuator_solution, error