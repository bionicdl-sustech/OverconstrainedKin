from math import pi
from abc import abstractmethod
import modern_robotics as mr
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.ERobot import ERobot
from spatialmath import SE3
import numpy as np
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
from typing import Tuple

class OpenChainModel(Robot):
    '''
        Reference:
            https://github.com/olmerg/rtb_serial_robot/blob/main/planar_3dof/Planar3DOF.py
    '''
    def __init__(self, open_chain_urdf_path, tld=CURRENT_DIR):

        args = super().URDF_read(
            open_chain_urdf_path,
            tld=tld
        )
        
        super().__init__(
            args[0],
            name=args[1]
        )

class BaseKinematics:
    def __init__(
        self,
        span_tree_urdf_path: str,
        span_tree_urdf_tld: str = CURRENT_DIR,
        screw_hip: np.ndarray | None = None,
        screw_thigh: np.ndarray | None = None,
        screw_knee: np.ndarray | None = None,
        M0: np.ndarray | None = None,
    ):
        """Base Kinematics class for overconstrained robots, which provides basic kinematic functionalities.

        This class is designed to be extended by specific kinematic models, such as BennettKinematics.
        It provides methods for forward and inverse kinematics, as well as joint position and velocity conversions.

        However, this class does not implement of conversion from closed-loop joint positions to spanning tree joint positions.

        Two backends are supported for the kinematics calculations:
            1. Modern Robotics (MR) for screw theory based kinematics.
            2. RTB (Robotics Toolbox) for analytical kinematics.

        For MR backend to work, the screw axes of the joints and initial pose must be provided.

        Args:
            span_tree_urdf_path (str): urdf path of the spanning tree (open chain) model
            span_tree_urdf_tld (str, optional): the parent directory of the urdf file (the folder which contains the .urdf). Defaults to CURRENT_DIR.
            screw_hip (np.ndarray | None, optional): the screw axis of the hip joint in the spanning tree model. Defaults to None.
            screw_thigh (np.ndarray | None, optional): the screw axis of the thigh joint in the spanning tree model. Defaults to None.
            screw_knee (np.ndarray | None, optional): the screw axis of the knee joint in the spanning tree model. Defaults to None.
            M0 (np.ndarray | None, optional): Pose of the end-effector (foot) in initial configuration. Defaults to None.
        """
        # Kinematics based on screw theory
        self.use_screw_kinematics = screw_hip is not None and screw_thigh is not None and screw_knee is not None
        if self.use_screw_kinematics:
            self._screw_list_span_tree = np.zeros((6,3)) # Screw axes in base frame
            self._screw_list_span_tree[:, 0] = screw_hip  # Screw for hip
            self._screw_list_span_tree[:, 1] = screw_thigh  # Screw for thigh
            self._screw_list_span_tree[:, 2] = screw_knee  # Screw for knee
            self._M0 = M0

            # Calculate screw axes in foot frame (body screw axes)
            self._body_screw_list_span_tree = np.zeros((6,3))
            self._body_screw_list_span_tree[:, 0] = mr.Adjoint(mr.TransInv(M0)) @ screw_hip
            self._body_screw_list_span_tree[:, 1] = mr.Adjoint(mr.TransInv(M0)) @ screw_thigh
            self._body_screw_list_span_tree[:, 2] = mr.Adjoint(mr.TransInv(M0)) @ screw_knee

        self._span_tree_urdf_path = span_tree_urdf_path

        # Load the URDF model
        self._span_tree_robot_model = OpenChainModel(self._span_tree_urdf_path, tld=span_tree_urdf_tld)
        print(self._span_tree_robot_model)
    
    @abstractmethod
    def span_tree_joint_pos_from_actuator(self, actuator_joint_pos: np.ndarray) -> np.ndarray:
        '''Convert actuator joint positions (hip, thigh, calf) to span tree joint positions (hip, thigh, knee).
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: Joint positions of the span tree joints (hip, thigh, knee).
        '''
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @abstractmethod
    def actuator_joint_pos_from_span_tree(self, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''Convert span tree joint positions (hip, thigh, knee) to actuator joint positions (hip, thigh, calf).
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: Joint positions of the actuator joints (hip, thigh, calf).
        '''
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @abstractmethod
    def span_tree_joint_vel_from_actuator(self, actuator_joint_pos: np.ndarray, actuator_joint_vel: np.ndarray) -> np.ndarray:
        '''Convert actuator joint velocities (hip, thigh, calf) to span tree joint velocities (hip, thigh, knee).
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
            actuator_joint_vel: Joint velocities of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: Joint velocities of the span tree joints (hip, thigh, knee).
        '''
        raise NotImplementedError("This method should be implemented in the subclass.")

    @abstractmethod
    def actuator_joint_vel_from_span_tree(self, span_tree_joint_pos: np.ndarray, span_tree_joint_vel: np.ndarray) -> np.ndarray:
        '''Convert span tree joint velocities (hip, thigh, knee) to actuator joint velocities (hip, thigh, calf).
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
            span_tree_joint_vel: Joint velocities of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: Joint velocities of the actuator joints (hip, thigh, calf).
        '''
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    @abstractmethod
    def vel_mapping_actuator_from_span_tree(self, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''The mapping matrix converting span tree joint velocities (hip, thigh, knee) to actuator joint velocities (hip, thigh, calf).

        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee
            span_tree_joint_vel: Joint velocities of the span tree joints (hip, thigh, knee).

        Returns:
            np.ndarray: The mapping matrix from span tree joint velocities to actuator joint velocities.
        '''
        return np.eye(3) # Default to identity matrix

    def inverse_kinematics_span_tree(self, goal_pos_in_hip_frame, ik_func="ik_GN") -> Tuple[np.ndarray, Tuple]:
        '''Calculate the inverse kinematics of the span tree given a goal position in the hip frame.
        
        Args:
            goal_pos_in_hip_frame: Goal position in the hip frame.
            ik_func: The inverse kinematics function to use (default is "ik_GN").

        Returns:
            np.ndarray: The solution actuator joint positions (hip, thigh, calf) that achieve the goal position in the hip frame.
            tuple: A tuple containing success status, iterations, searches, and residuals.
        '''
        tep = SE3(goal_pos_in_hip_frame[0], goal_pos_in_hip_frame[1], goal_pos_in_hip_frame[2])
        assert hasattr(self._span_tree_robot_model, ik_func), f"IK function {ik_func} not found in robot model."
        func = getattr(self._span_tree_robot_model, ik_func)
        sol_joint_pos, success, iterations, searches, residual = func(
            tep,
            mask=[1, 1, 1, 0, 0, 0],
        )
        
        return np.array(sol_joint_pos), (success, iterations, searches, residual)
    
    def inverse_kinematics(self, goal_pos_in_hip_frame, ik_func="ik_GN") -> Tuple[np.ndarray, Tuple]:
        '''Calculate the inverse kinematics of the robot given a goal position in the hip frame.
        
        Args:
            goal_pos_in_hip_frame: Goal position in the hip frame.
            ik_func: The inverse kinematics function to use (default is "ik_GN").

        Returns:
            np.ndarray: The solution actuator joint positions (hip, thigh, calf) that achieve the goal position in the hip frame.
            tuple: A tuple containing success status, iterations, searches, and residuals.
        '''
        sol_joint_pos, info_tuple = self.inverse_kinematics_span_tree(goal_pos_in_hip_frame, ik_func)
        actuator_joint_pos = self.actuator_joint_pos_from_span_tree(sol_joint_pos)
        return actuator_joint_pos, info_tuple

    def forward_kinematics_span_tree(self, span_tree_joint_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Calculate the forward kinematics of the span tree given the hip, thigh, and knee positions (span tree joint positions).
        
        Args:
            span_tree_joint_pos: Joint positions of span tree joints (hip, thigh, knee)
        '''
        T = mr.FKinSpace(self._M0, self._screw_list_span_tree, span_tree_joint_pos)
        return T[:3, 3], T
    
    def forward_kinematics_span_tree_rtb(self, span_tree_joint_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Calculate the forward kinematics of the span tree given the hip, thigh, and knee positions (span tree joint positions).
        
        Args:
            span_tree_joint_pos: Joint positions of span tree joints (hip, thigh, knee)
        '''
        T = self._span_tree_robot_model.fkine(span_tree_joint_pos).A
        return T[:3, 3], T
    
    def forward_kinematics(self, actuator_joint_pos) -> Tuple[np.ndarray, np.ndarray]:
        '''Calculate the forward kinematics of the robot given the hip, thigh, and calf positions (actuator joint positions).
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf)
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.forward_kinematics_span_tree(span_tree_joint_pos)
    
    def forward_kinematics_rtb(self, actuator_joint_pos) -> Tuple[np.ndarray, np.ndarray]:
        '''Calculate the forward kinematics of the robot given the hip, thigh, and calf positions (actuator joint positions).
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf)
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.forward_kinematics_span_tree_rtb(span_tree_joint_pos)
    
    def visualize_span_tree(self, span_tree_joint_pos: np.ndarray, backend="swift", block=False):
        '''Visualize the span tree given the joint positions of the span tree.
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        '''
        self._span_tree_robot_model.plot(
            q=span_tree_joint_pos,
            block=block,
            backend=backend,
        )

    def visualize(self, actuator_joint_pos: np.ndarray, backend="swift", block=False):
        '''Visualize the robot given the joint positions of the actuator joints.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        self.visualize_span_tree(span_tree_joint_pos, backend=backend, block=block)

    def space_jacobian_span_tree(self, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the space Jacobian of the span tree given the joint positions of the span tree.
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: The space Jacobian of the span tree.
        '''
        return mr.JacobianSpace(self._screw_list_span_tree, span_tree_joint_pos)
    
    def space_jacobian(self, actuator_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the space Jacobian of the robot given the joint positions of the actuator joints.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The space Jacobian of the robot.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.space_jacobian_span_tree(span_tree_joint_pos)
    
    def body_jacobian_span_tree(self, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the body Jacobian of the span tree given the joint positions of the span tree.
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: The body Jacobian of the span tree.
        '''
        return mr.JacobianBody(self._body_screw_list_span_tree, span_tree_joint_pos)
    
    def body_jacobian(self, actuator_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the body Jacobian of the robot given the joint positions of the actuator joints.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The body Jacobian of the robot.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.body_jacobian_span_tree(span_tree_joint_pos)
    
    def analytic_linear_jacobian_span_tree(self, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the linear velocity part of the analytic Jacobian of the span tree given the
        joint positions of the span tree.
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: The analytic Jacobian of the span tree.
        '''
        linear_jacobian = self.body_jacobian_span_tree(span_tree_joint_pos)[3:, :]
        Tsb = self.forward_kinematics_span_tree(span_tree_joint_pos)[1]
        Rsb = Tsb[:3, :3]
        return Rsb @ linear_jacobian
    
    def analytic_linear_jacobian(self, actuator_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the linear velocity part of the analytic Jacobian of the robot given the
        joint positions of the actuator joints.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The analytic Jacobian of the robot.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.analytic_linear_jacobian_span_tree(span_tree_joint_pos)
    
    def analytic_linear_jacobian_span_tree_rtb(self, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the linear velocity part of the analytic Jacobian of the span tree given the
        joint positions of the span tree using RTB.
        
        Args:
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: The analytic Jacobian of the span tree.
        '''
        linear_jacobian = self._span_tree_robot_model.jacob0_analytical(span_tree_joint_pos)[:3, :]
        return linear_jacobian
    
    def analytic_linear_jacobian_rtb(self, actuator_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the linear velocity part of the analytic Jacobian of the robot given the
        joint positions of the actuator joints using RTB.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The analytic Jacobian of the robot.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.analytic_linear_jacobian_span_tree_rtb(span_tree_joint_pos)
    
    def forward_velocity_kinematics(self, actuator_joint_pos: np.ndarray, actuator_joint_vel: np.ndarray) -> np.ndarray:
        '''Calculate the foot's linear velocity given the actuator joint positions and velocities of the actuator joints.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
            actuator_joint_vel: Joint velocities of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The end-effector linear and angular velocities in the base frame.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        span_tree_joint_vel = self.span_tree_joint_vel_from_actuator(actuator_joint_pos, actuator_joint_vel)
        J = self.analytic_linear_jacobian_span_tree(span_tree_joint_pos)
        return J @ span_tree_joint_vel
    
    def forward_velocity_kinematics_rtb(self, actuator_joint_pos: np.ndarray, actuator_joint_vel: np.ndarray) -> np.ndarray:
        '''Calculate the foot's linear velocity given the actuator joint positions and velocities of the actuator joints using RTB.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
            actuator_joint_vel: Joint velocities of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The end-effector linear and angular velocities in the base frame.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        span_tree_joint_vel = self.span_tree_joint_vel_from_actuator(actuator_joint_pos, actuator_joint_vel)
        J = self.analytic_linear_jacobian_span_tree_rtb(span_tree_joint_pos)
        return J @ span_tree_joint_vel
    
    def forward_statics(self, actuator_joint_pos: np.ndarray, actuator_torques: np.ndarray) -> np.ndarray:
        '''Calculate the foot's linear velocity given the actuator joint positions and torques of the actuator joints.
        
        Args:
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
            actuator_torques: Joint torques of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The end-effector linear and angular velocities in the base frame.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        J = self.analytic_linear_jacobian_span_tree(span_tree_joint_pos)
        return np.linalg.inv(J).T @ self.vel_mapping_actuator_from_span_tree(span_tree_joint_pos).T @ actuator_torques 

    def inverse_velocity_kinematics_span_tree(self, goal_vel_in_hip_frame: np.ndarray, span_tree_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the inverse velocity kinematics of the span tree given a goal velocity in the hip frame.
        
        Args:
            goal_vel_in_hip_frame: Goal velocity in the hip frame.
            span_tree_joint_pos: Joint positions of the span tree joints (hip, thigh, knee).
        
        Returns:
            np.ndarray: The solution span tree joint velocities (hip, thigh, calf) that achieve the goal velocity in the hip frame.
        '''
        J = self.analytic_linear_jacobian_span_tree(span_tree_joint_pos)
        return np.linalg.inv(J) @ goal_vel_in_hip_frame
    
    def inverse_velocity_kinematics(self, goal_vel_in_hip_frame: np.ndarray, actuator_joint_pos: np.ndarray) -> np.ndarray:
        '''Calculate the inverse velocity kinematics of the robot given a goal velocity in the hip frame.
        
        Args:
            goal_vel_in_hip_frame: Goal velocity in the hip frame.
            actuator_joint_pos: Joint positions of the actuator joints (hip, thigh, calf).
        
        Returns:
            np.ndarray: The solution actuator joint velocities (hip, thigh, calf) that achieve the goal velocity in the hip frame.
        '''
        span_tree_joint_pos = self.span_tree_joint_pos_from_actuator(actuator_joint_pos)
        return self.inverse_velocity_kinematics_span_tree(goal_vel_in_hip_frame, span_tree_joint_pos)