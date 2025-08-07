from .base_kinematics import BaseKinematics
from .utils.bennett_ik import GeneralBennettIK
import numpy as np
import math
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class MagicLimbBennettKin(BaseKinematics, GeneralBennettIK):
    def __init__(self):
        urdf_path = "MagicLimbRobotBennettFL.urdf"
        tld = os.path.join(CURRENT_DIR, 'urdfs', 'MagicLimbBennettRTB')

        # link_len = 0.162 # Theoretical length of the links
        foot_x0 = 0.377 # x position of foot in initial configuration
        foot_y0 = 0.018 # y position of foot in initial configuration
        foot_z0 = 0.07 # z position of foot in initial configuration

        screw_hip = np.array([1, 0, 0, 0, 0, 0])
        screw_thigh = np.array([0, 0, 1, 0, 0, 0])
        screw_knee_y = -math.sin(60* np.pi / 180.0)  # y component of the screw for the knee joint
        screw_knee_z = math.cos(60 * np.pi / 180.0)  # z component of the screw for the knee joint
        screw_knee_angular = np.array([0, screw_knee_y, screw_knee_z])  # Angular component of the screw for the knee joint
        q = np.array([0.162, 0, 0.0746188])  # A point on the rotation axis of the knee joint
        screw_knee_linear = np.cross(-screw_knee_angular, q)  # Linear component of the screw for the knee joint
        screw_knee = np.zeros(6)
        screw_knee[:3] = screw_knee_angular
        screw_knee[3:] = screw_knee_linear

        beta = 60 * np.pi / 180.0  # Bennett linkage beta
        alpha = 120 * np.pi / 180.0  # Bennett linkage alpha
        self.bennett_K = math.sin(0.5 * (beta + alpha)) / math.sin(0.5 * (beta - alpha))

        M0 = np.eye(4)
        M0[0, 3] = foot_x0
        M0[1, 3] = foot_y0
        M0[2, 3] = foot_z0

        # Constants for analytic ik
        self.l = 0.162 # Linkage length, same length for all links
        self.x0 = foot_x0 # x pos of foot in initial configuration
        self.y0 = foot_y0 # y pos of foot in initial configuration
        self.z0 = foot_z0 # z pos of foot in initial configuration
        self.zt = 0.0746188 # z position of the knee joint origin
        self.beta = beta

        self.A = -self.y0*math.cos(beta) - self.z0*math.sin(beta) + self.zt*math.sin(beta)
        self.B = -self.y0*math.sin(beta) + self.z0*math.cos(beta) - self.zt*math.cos(beta)
        self.C = (self.x0-self.l)
        self.D = self.y0*math.sin(beta)*math.cos(beta) - self.z0*math.cos(beta)**2 - self.zt*math.sin(beta)**2

        self.E = -self.zt*math.sin(beta)
        self.K = self.A**2 + self.B**2*math.sin(beta)**2 + self.C**2 + self.D**2 + self.l**2
        self.M = 2*(self.C*self.l + self.A*self.E)
        self.N = 2*(self.A*self.l - self.C*self.E)
        self.H = math.sqrt(self.M**2 + self.N**2)

        BaseKinematics.__init__(
            self,
            span_tree_urdf_path=urdf_path, 
            span_tree_urdf_tld=tld,
            screw_hip=screw_hip, 
            screw_thigh=screw_thigh, 
            screw_knee=screw_knee, 
            M0=M0,
        )

        GeneralBennettIK.__init__(self, self.l, self.x0, self.y0, self.z0, self.zt, self.beta,
            thigh_offset=0.0,
            knee_offset=0.0
        )

    # Implement mapping functions between actuator and span tree joint positions
    def span_tree_joint_pos_from_actuator(self, actuator_joint_pos):
        span_tree_joint_pos = np.zeros(3)
        # Hip joint position remains the same
        span_tree_joint_pos[0] = actuator_joint_pos[0]

        # Thigh joint position remains the same
        span_tree_joint_pos[1] = actuator_joint_pos[1]

        # Map the actuator joint position to span tree joint position
        span_tree_joint_pos[2] = 2 * math.atan(-self.bennett_K * math.tan((actuator_joint_pos[2]) / 2))

        return span_tree_joint_pos
    
    def actuator_joint_pos_from_span_tree(self, span_tree_joint_pos):
        actuator_joint_pos = np.zeros(3)
        # Hip joint position remains the same
        actuator_joint_pos[0] = span_tree_joint_pos[0]

        # Thigh joint position remains the same
        actuator_joint_pos[1] = span_tree_joint_pos[1]

        # For calf joint, add the offset
        actuator_joint_pos[2] = 2 * math.atan(-math.tan((span_tree_joint_pos[2]) / 2)/self.bennett_K)

        return actuator_joint_pos
    
    def actuator_joint_vel_from_span_tree(self, span_tree_joint_pos, span_tree_joint_vel):
        actuator_joint_vel = np.zeros(3)
        # Hip joint position remains the same
        actuator_joint_vel[0] = span_tree_joint_vel[0]

        # Thigh joint position remains the same
        actuator_joint_vel[1] = span_tree_joint_vel[1]

        # Map the span tree joint velocity to actuator joint velocity
        actuator_joint_vel[2] = -self.bennett_K/(self.bennett_K**2*math.cos(span_tree_joint_pos[2]/2)**2+math.sin(span_tree_joint_pos[2]/2)**2) * span_tree_joint_vel[2]

        return actuator_joint_vel
    
    def span_tree_joint_vel_from_actuator(self, actuator_joint_pos, actuator_joint_vel):
        span_tree_joint_vel = np.zeros(3)
        # Hip joint position remains the same
        span_tree_joint_vel[0] = actuator_joint_vel[0]

        # Thigh joint position remains the same
        span_tree_joint_vel[1] = actuator_joint_vel[1]

        # Map the actuator joint velocity to span tree joint velocity
        span_tree_joint_vel[2] = -self.bennett_K/(math.cos(actuator_joint_pos[2]/2)**2+self.bennett_K**2*math.sin(actuator_joint_pos[2]/2)**2) * actuator_joint_vel[2]

        return span_tree_joint_vel
    
    def vel_mapping_actuator_from_span_tree(self, span_tree_joint_pos):
        vel_mapping = np.eye(3)
        vel_mapping[2,2] = -self.bennett_K/(self.bennett_K**2*math.cos(span_tree_joint_pos[2]/2)**2+math.sin(span_tree_joint_pos[2]/2)**2)

        return vel_mapping

    def analytic_ik(self, goal_pos_in_hip_frame):
        span_tree_solution, success, ik_solutions, ik_errors = self.analytical_inverse_kinematics_span_tree(goal_pos_in_hip_frame)
        actuator_solution = self.actuator_joint_pos_from_span_tree(span_tree_solution)
        return actuator_solution, success, ik_solutions, ik_errors

    @staticmethod
    def get_theta(C1, C2, C3):
        """
        Calculate value of theta given C1, C2, C3 in form of :

        C1*cos(theta) + C2*sin(theta) + C3 = 0

        Two solutions are possible:
        theta1 = 2atan2(C2 +/- sqrt(C2**2 - C3**2 + C1**2), C1 - C3)
        """
        theta1 = 2 * math.atan2(C2 + math.sqrt(C2**2 - C3**2 + C1**2), C1 - C3)
        theta2 = 2 * math.atan2(C2 - math.sqrt(C2**2 - C3**2 + C1**2), C1 - C3)

        return theta1, theta2