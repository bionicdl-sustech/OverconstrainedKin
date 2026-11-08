import torch
from dataclasses import dataclass, asdict
from typing import Optional, List, Literal

from ov_kin.base_kinematics import GeneralBennettKinSolver, GeneralSphericalKinSolver

@dataclass
class GeneralBennettKinCfg:
    l: float # Linkage length between thigh and knee joint axis
    x0: float # x pos of foot in initial configuration
    y0: float # y pos of foot in initial configuration
    z0: float # z pos of foot in initial configuration
    zt: float # z position of the knee joint origin
    beta: float # Bennett linkage beta in radians
    alpha: float # Bennett linkage alpha in radians

    thigh_offset: float = 0.0 # Offset of the thigh joint axis from the knee joint axis along the knee joint axis
    knee_offset: float = 0.0 # Offset of the knee joint axis from the hip joint axis along the hip joint axis

@dataclass
class GeneralSphericalKinCfg:
    h: float # Linkage length between thigh and knee joint axis
    x0: float # x pos of foot in initial configuration
    y0: float # y pos of foot in initial configuration
    z0: float # z pos of foot in initial configuration
    alpha: float # Bennett linkage alpha in radians

    thigh_offset: float = 0.0 # Offset of the thigh joint axis from the knee joint axis along the knee joint axis
    knee_offset: float = 0.0 # Offset of the knee joint axis from the hip joint axis along the hip joint axis
    model_knee_offset: float = 0.0 # Offset of the knee joint axis from the hip joint axis along the hip joint axis used in the kinematic model

@dataclass
class RobotCfg:
    length: float # Length of the robot body, i.e. distance between the thigh/calf joints of the front and rear limbs
    width: float # Width of the robot body, i.e. distance between the hip joints of the left and right limbs

    actuator_signs_list: List[Literal[-1, 1]] # List of joint signs for each joint in the limb
    foot_pos_signs_list: List[Literal[-1, 1]] # List of foot position signs for each joint in the limb

class OvRobotKinSolver:
    def __init__(
        self, 
        kin_cfg: GeneralBennettKinCfg | GeneralSphericalKinCfg,
        robot_cfg: RobotCfg,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        self._kin_cfg = kin_cfg
        self._robot_cfg = robot_cfg
        self._device = device
        self._dtype = dtype
        self._output_dtype = output_dtype
        
        # determine if the solver is spherical or not
        self._is_spherical = isinstance(kin_cfg, GeneralSphericalKinCfg)

        # Create solver
        if self._is_spherical:
            self._kin_solver = GeneralSphericalKinSolver(
                **asdict(kin_cfg),
                device=device, 
                dtype=dtype, 
                output_dtype=output_dtype
            )
        else:
            self._kin_solver = GeneralBennettKinSolver(
                **asdict(kin_cfg),
                device=device, 
                dtype=dtype, 
                output_dtype=output_dtype
            )

        # Check that the number of limbs is consistent with the robot configuration
        assert len(robot_cfg.actuator_signs_list) == len(robot_cfg.foot_pos_signs_list), "The number of actuator signs must match the number of foot position signs."

        # Check if length of actuator_signs_list and foot_pos_signs_list are 3
        for actuator_sign, foot_pos_sign in zip(robot_cfg.actuator_signs_list, robot_cfg.foot_pos_signs_list):
            assert len(actuator_sign) == 3, "Each actuator sign list must have length 3."
            assert len(foot_pos_sign) == 3, "Each foot position sign list must have length 3."

        self.__num_limbs = len(self._robot_cfg.actuator_signs_list)

        self.__single_batch_actuator_signs = torch.tensor(self._robot_cfg.actuator_signs_list, device=self._device, dtype=self._dtype).unsqueeze(0)
        self.__single_batch_foot_pos_signs = torch.tensor(self._robot_cfg.foot_pos_signs_list, device=self._device, dtype=self._dtype).unsqueeze(0)
        self.__foot_pos_offset = torch.tensor([[self._robot_cfg.length / 2.0, self._robot_cfg.width / 2.0, 0.0]], device=self._device, dtype=self._dtype).unsqueeze(0).repeat(1, self.__num_limbs, 1)
        # The foot position offset is the same for all limbs, but the signs are different
        self.__foot_pos_offset = self.__foot_pos_offset * self.__single_batch_foot_pos_signs

    def _actuator_pos_to_batch(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert actuator positions to a batch of actuator positions of all limbs.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size*num_limbs, 3)
        """
        # Mirror actuator positions using the actuator signs
        actuator_pos = actuator_pos * self.__single_batch_actuator_signs

        # Reshape the actuator pos to (batch_size * num_limbs, 3) for the kinematic solver
        actuator_pos = actuator_pos.view(-1, 3)
        return actuator_pos
    
    def _foot_pos_to_batch(self, foot_pos: torch.Tensor) -> torch.Tensor:
        """
        Convert foot positions to a batch of foot positions of all limbs.

        Args:
            foot_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the foot positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size*num_limbs, 3)
        """
        # Remove offset from foot positions
        foot_pos = foot_pos - self.__foot_pos_offset

        # Apply foot position signs
        foot_pos = foot_pos * self.__single_batch_foot_pos_signs

        # Reshape the actuator pos to (batch_size * num_limbs, 3) for the kinematic solver
        foot_pos = foot_pos.view(-1, 3)
        return foot_pos

    def _foot_vel_force_to_batch(self, foot_vel_force: torch.Tensor) -> torch.Tensor:
        """
        Convert foot velocities or forces to a batch of foot velocities or forces of all limbs.

        Args:
            foot_vel_force (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the foot velocities or forces for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size*num_limbs, 3)
        """
        # Apply foot velocity/force signs
        foot_vel_force = foot_vel_force * self.__single_batch_foot_pos_signs # Same as foot_pos_signs_list

        # Reshape the actuator pos to (batch_size * num_limbs, 3) for the kinematic solver
        foot_vel_force = foot_vel_force.view(-1, 3)
        return foot_vel_force

    def _recover_batched_foot_pos(self, foot_pos: torch.Tensor) -> torch.Tensor:
        """
        Recover the original shape of foot positions from a batch of foot positions of all limbs.

        Args:
            foot_pos (torch.Tensor): A tensor of shape (batch_size*num_limbs, 3) representing the foot positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the foot positions for each limb.
        """
        reshaped_foot_pos_tensor = foot_pos.view(-1, self.__num_limbs, 3)

        # Apply foot position signs
        reshaped_foot_pos_tensor = reshaped_foot_pos_tensor * self.__single_batch_foot_pos_signs

        # Add offset to foot positions
        reshaped_foot_pos_tensor = reshaped_foot_pos_tensor + self.__foot_pos_offset

        return reshaped_foot_pos_tensor.to(dtype=self._dtype)

    def _recover_batched_foot_vel_force(self, foot_vel_force: torch.Tensor) -> torch.Tensor:
        """
        Recover the original shape of foot velocities or forces from a batch of foot velocities or forces of all limbs.

        Args:
            foot_vel_force (torch.Tensor): A tensor of shape (batch_size*num_limbs, 3) representing the foot velocities or forces for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the foot velocities or forces for each limb.
        """
        reshaped_foot_vel_force_tensor = foot_vel_force.view(-1, self.__num_limbs, 3)

        # Apply foot velocity/force signs
        reshaped_foot_vel_force_tensor = reshaped_foot_vel_force_tensor * self.__single_batch_foot_pos_signs

        return reshaped_foot_vel_force_tensor.to(dtype=self._dtype)

    def _recover_batched_actuator_pos(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Recover the original shape of actuator positions from a batch of actuator positions of all limbs.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size*num_limbs, 3) representing the actuator positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.
        """
        reshaped_actuator_pos_tensor = actuator_pos.view(-1, self.__num_limbs, 3)

        # Apply actuator position signs
        reshaped_actuator_pos_tensor = reshaped_actuator_pos_tensor * self.__single_batch_actuator_signs

        return reshaped_actuator_pos_tensor.to(dtype=self._dtype)

    def knee_pos_from_actuator(self, actuator_pos: torch.Tensor, use_urdf_convention: bool = True) -> torch.Tensor:
        """
        Get knee joint positions from actuator positions

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.
            use_urdf_convention (bool): urdf file assumes all knee joint positions to be non-negative (which does not make sense but we 
            keep it as it is). If this is set to True, no sign post-processing will be done. 

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 1) representing the knee joint positions for each limb.
        """
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)
        batched_knee_pos = self._kin_solver.knee_pos_from_actuator(batched_actuator_pos)
        if not use_urdf_convention:
            knee_pos = self._recover_batched_actuator_pos(batched_knee_pos)
        else:
            knee_pos = torch.abs(batched_knee_pos)
        return knee_pos

    def fk(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward kinematics of the robot given the actuator positions.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the foot positions for each limb.
        """
        # Check that actuator_pos has the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."

        # Reshape the actuator pos to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)

        # Compute foot positions using the kinematic solver
        foot_positions = self._kin_solver.forward_kinematics(batched_actuator_pos)

        # Recover the original shape of foot_positions to (batch_size, num_limbs, 3)
        foot_positions = self._recover_batched_foot_pos(foot_positions)

        return foot_positions.to(dtype=self._output_dtype)

    def ik(self, foot_pos: torch.Tensor, ref_actuator_pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the inverse kinematics of the robot given the foot positions and reference actuator positions.

        Args:
            foot_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the foot positions for each limb.
            ref_actuator_pos (torch.Tensor | None): A tensor of shape (batch_size, num_limbs, 3) representing the reference actuator positions for each limb.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Computed actuator positions (batch_size, num_limbs, 3),
            IK errors (batch_size, num_limbs), and out_of_workspace flags (batch_size, num_limbs).
        """
        # Check that tensors have the correct shape
        assert foot_pos.shape[-1] == 3, "The last dimension of foot_pos must be 3."
        assert foot_pos.shape[1] == self.__num_limbs, f"The second dimension of foot_pos must be {self.__num_limbs}."
        if ref_actuator_pos is not None:
            assert ref_actuator_pos.shape[-1] == 3, "The last dimension of ref_actuator_pos must be 3."
            assert ref_actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of ref_actuator_pos must be {self.__num_limbs}."

        # Reshape the tensors to (batch_size * num_limbs, 3) for the kinematic solver
        batched_foot_pos = self._foot_pos_to_batch(foot_pos)
        batched_ref_actuator_pos = self._actuator_pos_to_batch(ref_actuator_pos) if ref_actuator_pos is not None else None

        # Compute actuator positions using the kinematic solver
        actuator_positions, ik_errors, out_of_workspace = self._kin_solver.inverse_kinematics(batched_foot_pos, batched_ref_actuator_pos)

        # Recover the original shape of actuator_positions to (batch_size, num_limbs, 3)
        actuator_positions = self._recover_batched_actuator_pos(actuator_positions)

        return (
            actuator_positions.to(dtype=self._output_dtype),
            ik_errors.view(-1, self.__num_limbs).to(dtype=self._output_dtype),
            out_of_workspace.view(-1, self.__num_limbs)
        )

    def J(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute the analytical Jacobian of the robot given the actuator positions. 
        IMPORTANT NOTE: analytical Jacobian is not the same as the geometric Jacobian!

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3, 3) representing the Jacobian for each limb.
        """
        # Check that actuator_pos has the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."

        # Reshape the actuator pos to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)

        # Compute Jacobian using the kinematic solver
        jacobian = self._kin_solver.actuator_jacobian(batched_actuator_pos)

        # Recover the original shape of jacobian to (batch_size, num_limbs, 3, 3)
        jacobian = jacobian.view(-1, self.__num_limbs, 3, 3)

        return jacobian.to(dtype=self._output_dtype)

    def J_inverse(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of the analytical Jacobian of the robot given the actuator positions. 
        IMPORTANT NOTE: analytical Jacobian is not the same as the geometric Jacobian!

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3, 3) representing the inverse Jacobian for each limb.
        """
        # Check that actuator_pos has the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."

        # Reshape the actuator pos to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)

        # Compute inverse Jacobian using the kinematic solver
        jacobian_inverse = self._kin_solver.actuator_jacobian_inverse(batched_actuator_pos)

        # Recover the original shape of jacobian_inverse to (batch_size, num_limbs, 3, 3)
        jacobian_inverse = jacobian_inverse.view(-1, self.__num_limbs, 3, 3)

        return jacobian_inverse.to(dtype=self._output_dtype)

    def forward_velocity_kinematics(self, actuator_pos: torch.Tensor, actuator_vel: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward velocity kinematics of the robot given the actuator positions and velocities.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.
            actuator_vel (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator velocities for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the forward velocity kinematics for each limb.
        """
        # Check that actuator_pos and actuator_vel have the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."
        assert actuator_vel.shape[-1] == 3, "The last dimension of actuator_vel must be 3."
        assert actuator_vel.shape[1] == self.__num_limbs, f"The second dimension of actuator_vel must be {self.__num_limbs}."

        # Reshape the actuator pos and vel to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)
        batched_actuator_vel = self._actuator_pos_to_batch(actuator_vel) # Same conversion as actuator_pos

        # Compute forward velocity kinematics using the kinematic solver
        forward_velocity = self._kin_solver.forward_velocity_kinematics(batched_actuator_pos, batched_actuator_vel)

        # Recover the original shape of forward_velocity to (batch_size, num_limbs, 3)
        forward_velocity = self._recover_batched_foot_vel_force(forward_velocity)

        return forward_velocity.to(dtype=self._output_dtype)

    def inverse_velocity_kinematics(self, actuator_pos: torch.Tensor, foot_vel: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse velocity kinematics of the robot given the actuator positions and foot velocities.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.
            foot_vel (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the foot velocities for each limb.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the inverse velocity kinematics for each limb.
        """
        # Check that actuator_pos and foot_vel have the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."
        assert foot_vel.shape[-1] == 3, "The last dimension of foot_vel must be 3."
        assert foot_vel.shape[1] == self.__num_limbs, f"The second dimension of foot_vel must be {self.__num_limbs}."

        # Reshape the actuator pos and foot vel to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)
        batched_foot_vel = self._foot_vel_force_to_batch(foot_vel)

        # Compute inverse velocity kinematics using the kinematic solver (goal_foot_vel first, actuator_joint_pos second)
        inverse_velocity = self._kin_solver.inverse_velocity_kinematics(batched_foot_vel, batched_actuator_pos)

        # Recover the original shape of actuator velocities to (batch_size, num_limbs, 3)
        inverse_velocity = self._recover_batched_actuator_pos(inverse_velocity)

        return inverse_velocity.to(dtype=self._output_dtype)

    def forward_statics(self, actuator_pos: torch.Tensor, actuator_force: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward statics of the robot given the actuator positions and forces.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.
            actuator_force (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator forces for each limb.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the forward statics for each limb.
        """
        # Check that actuator_pos and actuator_force have the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."
        assert actuator_force.shape[-1] == 3, "The last dimension of actuator_force must be 3."
        assert actuator_force.shape[1] == self.__num_limbs, f"The second dimension of actuator_force must be {self.__num_limbs}."

        # Reshape the actuator pos and force to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)
        batched_actuator_force = self._actuator_pos_to_batch(actuator_force) # Same conversion as actuator_pos

        # Compute forward statics using the kinematic solver
        forward_statics = self._kin_solver.forward_statics(batched_actuator_pos, batched_actuator_force)

        # Recover the original shape of forward_statics to (batch_size, num_limbs, 3)
        forward_statics = self._recover_batched_foot_vel_force(forward_statics)

        return forward_statics.to(dtype=self._output_dtype)

    def inverse_statics(self, actuator_pos: torch.Tensor, foot_force: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse statics of the robot given the actuator positions and foot forces.

        Args:
            actuator_pos (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the actuator positions for each limb.
            foot_force (torch.Tensor): A tensor of shape (batch_size, num_limbs, 3) representing the foot forces for each limb.
        Returns:
            torch.Tensor: A tensor of shape (batch_size, num_limbs, 3) representing the inverse statics for each limb.
        """
        # Check that actuator_pos and foot_force have the correct shape
        assert actuator_pos.shape[-1] == 3, "The last dimension of actuator_pos must be 3."
        assert actuator_pos.shape[1] == self.__num_limbs, f"The second dimension of actuator_pos must be {self.__num_limbs}."
        assert foot_force.shape[-1] == 3, "The last dimension of foot_force must be 3."
        assert foot_force.shape[1] == self.__num_limbs, f"The second dimension of foot_force must be {self.__num_limbs}."

        # Reshape the actuator pos and foot force to (batch_size * num_limbs, 3) for the kinematic solver
        batched_actuator_pos = self._actuator_pos_to_batch(actuator_pos)
        batched_foot_force = self._foot_vel_force_to_batch(foot_force)

        # Compute inverse statics using the kinematic solver
        inverse_statics = self._kin_solver.inverse_statics(batched_actuator_pos, batched_foot_force)

        # Recover the original shape of actuator torques to (batch_size, num_limbs, 3)
        inverse_statics = self._recover_batched_actuator_pos(inverse_statics)

        return inverse_statics.to(dtype=self._output_dtype)
