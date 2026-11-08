import torch
from math import pi, sqrt, atan, sin, cos

def _skew_symmetric(v: torch.Tensor):
    """
    Calculate the skew symmetric matrix of a vector

    Args:
        v (torch.Tensor): vector in shape of (n, 3)
    """
    # # normalize the vector
    # v = v / torch.norm(v, dim=1, keepdim=True)
    batch_size = v.size(0)
    zeros = torch.zeros(batch_size, device=v.device)
    matrix = torch.stack((zeros, -v[:, 2], v[:, 1], v[:, 2], zeros, -v[:, 0], -v[:, 1], v[:, 0], zeros), dim=1)
    return matrix.view(batch_size, 3, 3)

def exp_rotation(w: torch.Tensor, theta: torch.Tensor):
    w_hat = _skew_symmetric(w)
    w_hat_square = torch.bmm(w_hat, w_hat)

    I = torch.eye(3).unsqueeze(0).repeat(w.size(0), 1, 1).to(w.device)

    R = (
        I
        + torch.sin(theta).unsqueeze(-1).unsqueeze(-1) * w_hat
        + (1 - torch.cos(theta).unsqueeze(-1).unsqueeze(-1)) * w_hat_square
    )
    return R

def exp_motion(s: torch.Tensor, theta: torch.Tensor):
    """
    Calculate the motion matrix given the screw axis and angle

    Args:
        s (torch.Tensor): screw axis in shape of (n, 6)
        theta (torch.Tensor): angle in shape of (n,)
    """
    w = s[:, :3]
    v = s[:, 3:]
    w = w / torch.norm(w, dim=-1, keepdim=True)
    R = exp_rotation(w, theta)
    w_skew = _skew_symmetric(w)
    w_skew_square = torch.bmm(w_skew, w_skew)
    theta_mat = theta.reshape(-1, 1, 1).repeat(1, 3, 3)
    I_theta = torch.eye(3, device=s.device).expand_as(w_skew) * theta_mat
    one_minus_cos_theta = 1 - torch.cos(theta_mat)
    theta_minus_sin_theta = theta_mat - torch.sin(theta_mat)

    t = torch.bmm(I_theta + one_minus_cos_theta * w_skew + theta_minus_sin_theta * w_skew_square, v.reshape(-1, 3, 1))

    transform = torch.cat(
        (
            torch.cat((R, t), dim=-1),
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=w.device).view(1, 1, -1).expand(theta.shape[0], 1, -1),
        ),
        dim=-2,
    )
    return transform

def adjoint_transform(T: torch.Tensor):
    """
    Calculate the adjoint transformation matrix given the transformation matrix

    Args:
        T (torch.Tensor): transformation matrix in shape of (n, 4, 4)
    """
    R = T[:, :3, :3]
    p = T[:, :3, 3]
    p_skew = _skew_symmetric(p)
    p_cross_R = torch.bmm(p_skew, R)
    adj_T = torch.zeros(T.size(0), 6, 6, device=T.device, dtype=T.dtype)
    adj_T[:, 0:3, 0:3] = R
    adj_T[:, 3:6, 0:3] = p_cross_R
    adj_T[:, 3:6, 3:6] = R
    return adj_T

class GeneralBennettKinSolver:
    def __init__(
            self,
            l, x0, y0, z0, zt, beta, alpha,
            thigh_offset=0.0, knee_offset=0.0,
            device="cuda",
            dtype=torch.float64,
            output_dtype=torch.float32
        ):
        self.thigh_offset = thigh_offset # Motor thigh angular offset in radians
        self.knee_offset = knee_offset # Motor knee angular offset in radians

        self.l = l  # Linkage length, same length for all links
        self.x0 = x0  # x pos of foot in initial configuration
        self.y0 = y0  # y pos of foot in initial configuration
        self.z0 = z0  # z pos of foot in initial configuration
        self.zt = zt  # z position of the knee joint origin
        self.beta = beta  # Bennett linkage beta in radians
        self.alpha = alpha  # Bennett linkage alpha in radians
        self.dtype = dtype
        self.output_dtype = output_dtype

        self.thigh_offset = thigh_offset  # Thigh motor offset in radians
        self.knee_offset = knee_offset  # Knee motor offset in radians

        self.sin_beta = sin(self.beta)
        self.cos_beta = cos(self.beta)

        self.A = -self.y0*cos(self.beta) - self.z0*sin(self.beta) + self.zt*sin(self.beta)
        self.B = -self.y0*sin(self.beta) + self.z0*cos(self.beta) - self.zt*cos(self.beta)
        self.C = (self.x0-self.l)
        self.D = self.y0*sin(self.beta)*cos(self.beta) - self.z0*cos(self.beta)**2 - self.zt*sin(self.beta)**2

        self.E = -self.zt*sin(self.beta)
        self.K = self.A**2 + self.B**2*sin(self.beta)**2 + self.C**2 + self.D**2 + self.l**2
        self.M = 2*(self.C*self.l + self.A*self.E)
        self.N = 2*(self.A*self.l - self.C*self.E)
        self.H = sqrt(self.M**2 + self.N**2)

        self.device = device

        # Get the transformation matrix for the initial configuration
        self._M0 = torch.eye(4, device=self.device, dtype=self.dtype)
        self._M0[0, 3] = self.x0
        self._M0[1, 3] = self.y0
        self._M0[2, 3] = self.z0

        # Get screw axis
        self.hip_axis = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        self.thigh_axis = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        self.knee_axis = torch.tensor([0.0, -self.sin_beta, self.cos_beta, 0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
        point_on_knee_axis = torch.tensor([self.l, 0.0, self.zt], device=self.device, dtype=self.dtype)
        self.knee_axis[3:6] = torch.cross(-self.knee_axis[:3], point_on_knee_axis, dim=0)

        # Calculate body screw axis
        self._M0_inv = torch.inverse(self._M0)
        self.hip_axis_b = adjoint_transform(self._M0_inv.unsqueeze(0)).squeeze(0) @ self.hip_axis
        self.thigh_axis_b = adjoint_transform(self._M0_inv.unsqueeze(0)).squeeze(0) @ self.thigh_axis
        self.knee_axis_b = adjoint_transform(self._M0_inv.unsqueeze(0)).squeeze(0) @ self.knee_axis

        # Calculate Bennett K
        self.bennett_K = sin(0.5 * (beta + alpha)) / sin(0.5 * (beta - alpha))

    '''Open Chain - Actuator Model Mapping Functions'''
    def actuator_pos_from_open_chain(self, open_chain_joint_pos: torch.Tensor) -> torch.Tensor:
        actuator_pos = open_chain_joint_pos.clone()
        # Convert open chain joint positions to cad open chain positions
        actuator_pos[:, 1] += self.thigh_offset
        actuator_pos[:, 2] += self.knee_offset
        # Convert cad open chain positions to actuator positions
        actuator_pos[:, 2] = 2 * torch.atan(-torch.tan((actuator_pos[:, 2]) / 2)/self.bennett_K)
        return actuator_pos
    
    def open_chain_pos_from_actuator(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        open_chain_pos = actuator_pos.clone()
        # Convert actuator positions to cad open chain positions
        open_chain_pos[:, 2] = 2 * torch.atan(-self.bennett_K * torch.tan((actuator_pos[:, 2]) / 2))
        # Convert cad open chain positions to open chain joint positions
        open_chain_pos[:, 1] -= self.thigh_offset
        open_chain_pos[:, 2] -= self.knee_offset
        return open_chain_pos

    def knee_pos_from_actuator(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        knee_pos = 2 * torch.atan(-self.bennett_K * torch.tan((actuator_pos[:, 2]) / 2))
        return knee_pos

    def velocity_mapping(self, actuator_joint_pos: torch.Tensor) -> torch.Tensor:
        """Calculate the velocity mapping from actuator joint velocities to open chain joint velocities.
        
        Args:
            actuator_joint_pos (torch.Tensor): Actuator joint positions of shape (N, 3).
        Returns:
            torch.Tensor: Velocity mapping matrix of shape (N, 3, 3).
        """
        velocity_mapping = torch.eye(3, 3, device=self.device, dtype=self.dtype).repeat(actuator_joint_pos.shape[0], 1, 1)
        motor_calf_joint_pos = actuator_joint_pos[:, 2]
        velocity_mapping[:, 2, 2] = -self.bennett_K/(torch.cos(motor_calf_joint_pos/2)**2+self.bennett_K**2*torch.sin(motor_calf_joint_pos/2)**2)
        return velocity_mapping

    '''Forward Kinematics Functions'''
    def forward_kinematics(self, actuator_joint_pos: torch.Tensor) -> torch.Tensor:
        open_chain_joint_pos = self.open_chain_pos_from_actuator(actuator_joint_pos)
        return self.forward_kinematics_open_chain(open_chain_joint_pos)
    
    def forward_kinematics_open_chain(self, open_chain_joint_pos: torch.Tensor) -> torch.Tensor:
        """Forward kinematics for the open chain joint positions.
        Args:
            open_chain_joint_pos (torch.Tensor): Open chain joint positions of shape (N, 3).
        Returns:
            torch.Tensor: Forward kinematics positions of shape (N, 3).
        """
        return self.forward_kinematics_open_chain_full(open_chain_joint_pos)[:, :3, 3]
    
    def forward_kinematics_open_chain_full(self, open_chain_joint_pos: torch.Tensor) -> torch.Tensor:
        """Forward kinematics for the open chain joint positions.
        Args:
            open_chain_joint_pos (torch.Tensor): Open chain joint positions of shape (N, 3).
        Returns:
            torch.Tensor: Forward kinematics transformation matrices of shape (N, 4, 4).
        """
        batch_dim = open_chain_joint_pos.shape[0]
        # Repeat the screw axes for the number of environments
        hip_axis = self.hip_axis.unsqueeze(0).repeat(batch_dim, 1)
        thigh_axis = self.thigh_axis.unsqueeze(0).repeat(batch_dim, 1)
        knee_axis = self.knee_axis.unsqueeze(0).repeat(batch_dim, 1)
        _M0 = self._M0.unsqueeze(0).repeat(batch_dim, 1, 1)

        T = exp_motion(hip_axis, open_chain_joint_pos[:, 0])
        T = torch.bmm(T, exp_motion(thigh_axis, open_chain_joint_pos[:, 1]))
        T = torch.bmm(T, exp_motion(knee_axis, open_chain_joint_pos[:, 2]))
        T = torch.bmm(T, _M0)
        return T

    '''Inverse Kinematics Functions'''
    def inverse_kinematics(
        self, goal_foot_pos: torch.Tensor, ref_actuator_pos: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref_open_chain_pos_tensor = self.open_chain_pos_from_actuator(ref_actuator_pos) if ref_actuator_pos is not None else None
        open_chain_joint_pos, ik_error, out_of_workspace = self.inverse_kinematics_open_chain(goal_foot_pos, ref_open_chain_pos_tensor)
        actuator_joint_pos = self.actuator_pos_from_open_chain(open_chain_joint_pos)
        return actuator_joint_pos, ik_error, out_of_workspace
    
    def inverse_kinematics_open_chain(
        self, pos_tensor: torch.Tensor, ref_pos_tensor: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = pos_tensor.shape[0]
        # There are totally 8 possible solutions
        ik_solutions = torch.zeros((batch_size, 8, 3), device=self.device, dtype=self.dtype)

        x = pos_tensor[:, 0:1]
        y = pos_tensor[:, 1:2]
        z = pos_tensor[:, 2:3]

        R = x**2 + y**2 + z**2
        S = R - self.K

        # 1. Check if the goal position is out of workspace in the radial/q3 dimension (Delta_q3 < 0)
        raw_mns_square_sum = self.M**2 + self.N**2 - S**2
        out_of_workspace_q3 = raw_mns_square_sum < -1e-6

        # Clamp S to [-H, H] and mns_square_sum to >= 0 so out-of-workspace targets project onto the boundary
        S_clamped = torch.clamp(S, min=-self.H, max=self.H)
        mns_square_sum = torch.clamp(self.M**2 + self.N**2 - S_clamped**2, min=0.0)
        sqrt_mns = torch.sqrt(mns_square_sum)

        # Avoid catastrophic cancellation regardless of the sign of self.N
        if self.N >= 0:
            q3_1 = 2 * torch.atan2(self.N + sqrt_mns, self.M + S_clamped)
            q3_2 = 2 * torch.atan2(S_clamped - self.M, self.N + sqrt_mns)
        else:
            q3_1 = 2 * torch.atan2(S_clamped - self.M, self.N - sqrt_mns)
            q3_2 = 2 * torch.atan2(self.N - sqrt_mns, self.M + S_clamped)

        ik_solutions[:, 0:4, 2] = q3_1.repeat(1, 4)
        ik_solutions[:, 4:8, 2] = q3_2.repeat(1, 4)

        out_of_workspace_branches = torch.zeros((batch_size, 8), device=self.device, dtype=torch.bool)

        for offset, q3 in [(0, q3_1), (4, q3_2)]:
            # Solve for q2 and q1
            P = self.C*torch.cos(q3) + self.A*torch.sin(q3) + self.l
            Q = self.A*self.cos_beta*torch.cos(q3) - \
                self.C*self.cos_beta*torch.sin(q3) + \
                self.B*self.sin_beta
            U = self.A*self.sin_beta*torch.cos(q3) - \
                self.C*self.sin_beta*torch.sin(q3) + \
                self.D
            square_sum = y**2 + z**2 - U**2
            out_of_workspace_q12 = (square_sum < -1e-6).squeeze(1)
            branch_out_of_ws = torch.logical_or(out_of_workspace_q3.squeeze(1), out_of_workspace_q12)
            out_of_workspace_branches[:, offset:offset + 4] = branch_out_of_ws.unsqueeze(1).repeat(1, 4)

            T = torch.sqrt(torch.clamp(square_sum, min=0.0))

            q2_1 = torch.atan2(Q * x - P * T, P * x + Q * T)
            q2_2 = torch.atan2(Q * x + P * T, P * x - Q * T)
            q1_1 = torch.atan2(U * y + T * z, T * y - U * z)
            q1_2 = torch.atan2(U * y - T * z, -T * y - U * z)

            pairs = [(q1_1, q2_1), (q1_1, q2_2), (q1_2, q2_1), (q1_2, q2_2)]
            for i, (q1_cand, q2_cand) in enumerate(pairs):
                ik_solutions[:, offset + i, 0:1] = q1_cand
                ik_solutions[:, offset + i, 1:2] = q2_cand

        # 4. Evaluate FK error on raw candidates
        fk_pos_tensor = self.forward_kinematics_open_chain(ik_solutions.view(-1, 3)).view(batch_size, 8, 3)
        ik_errors = torch.norm(fk_pos_tensor - pos_tensor.unsqueeze(1), dim=2)

        # 5. Select the best candidate
        if ref_pos_tensor is not None:
            ref_pos = ref_pos_tensor.to(dtype=self.dtype)
            min_ik_error, _ = torch.min(ik_errors, dim=1, keepdim=True)
            valid_fk_mask = ik_errors <= (min_ik_error + 1e-4)

            ang_diff = self.angle_diff(ik_solutions, ref_pos.unsqueeze(1))
            ref_pos_error = torch.norm(ang_diff, dim=2)

            masked_ref_pos_error = torch.where(
                valid_fk_mask,
                ref_pos_error,
                torch.full_like(ref_pos_error, float("inf"))
            )
            best_indices = torch.argmin(masked_ref_pos_error, dim=1)
        else:
            # When no ref_pos is provided, prefer physical knee solutions where sin(q3) <= 0 (i.e. q3 in [-pi, 0])
            within_q3_limit = torch.sin(ik_solutions[:, :, 2]) <= 1e-5
            penalized_ik_errors = torch.where(within_q3_limit, ik_errors, ik_errors + 100.0)
            best_indices = torch.argmin(penalized_ik_errors, dim=1)

        # Select the solution with the minimum error
        batch_indices = torch.arange(batch_size, device=self.device)
        best_solutions = ik_solutions[batch_indices, best_indices]
        best_errors = ik_errors[batch_indices, best_indices]
        out_of_workspace = torch.logical_or(
            out_of_workspace_branches[batch_indices, best_indices],
            best_errors > 1e-4
        )

        # 6. Convert ONLY the winning solution to the desired ranges at the very end
        best_solutions = self.format_joint_ranges(best_solutions)

        return best_solutions.to(dtype=self.output_dtype), best_errors.to(dtype=self.output_dtype), out_of_workspace
    
    def format_joint_ranges(self, q: torch.Tensor) -> torch.Tensor:
        """Converts winning joint angles to their desired canonical ranges (modulo 2*pi)."""
        q_out = q.clone()
        # 1. q1 in [-pi, pi]
        q_out[:, 0] = torch.atan2(torch.sin(q[:, 0]), torch.cos(q[:, 0]))
        # 2. q2 in [-thigh_offset, -thigh_offset + 2*pi)
        q_out[:, 1] = torch.remainder(q[:, 1] + self.thigh_offset, 2 * torch.pi) - self.thigh_offset
        # 3. q3 in [-pi, pi], mapping +pi to -pi so [-pi, 0] includes the -pi boundary
        q3_wrapped = torch.atan2(torch.sin(q[:, 2]), torch.cos(q[:, 2]))
        q_out[:, 2] = torch.where(q3_wrapped > torch.pi - 1e-7, q3_wrapped - 2 * torch.pi, q3_wrapped)
        return q_out

    @staticmethod
    def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Shortest angular difference in [-pi, pi] between two angles."""
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    '''Jacobian Functions'''
    def actuator_jacobian(self, actuator_joint_pos: torch.Tensor) -> torch.Tensor:
        open_chain_joint_pos = self.open_chain_pos_from_actuator(actuator_joint_pos)
        # Get Jacobian
        open_chain_jacobian = self.analytic_jacobian_open_chain(open_chain_joint_pos=open_chain_joint_pos)
        velocity_mapping = self.velocity_mapping(actuator_joint_pos=actuator_joint_pos)
        actuator_jacobian = torch.bmm(open_chain_jacobian, velocity_mapping)
        return actuator_jacobian

    def actuator_jacobian_inverse(self, actuator_joint_pos: torch.Tensor) -> torch.Tensor:
        actuator_jacobian = self.actuator_jacobian(actuator_joint_pos)
        actuator_jacobian_inverse = torch.linalg.inv(actuator_jacobian)
        return actuator_jacobian_inverse

    def body_jacobian_open_chain(self, open_chain_joint_pos: torch.Tensor, translation_only: bool=True) -> torch.Tensor:
        """Calculate the body Jacobian for the given open chain joint positions.
        
        Args:
            open_chain_joint_pos (torch.Tensor): Open chain joint positions of shape (N, 3).
        """
        body_jacobian = torch.zeros((open_chain_joint_pos.shape[0], 6, 3), device=self.device, dtype=self.dtype)
        body_jacobian[:, :, 2] = self.knee_axis_b.unsqueeze(0).repeat(open_chain_joint_pos.shape[0], 1)
        T_theta3 = exp_motion(self.knee_axis_b.unsqueeze(0).repeat(open_chain_joint_pos.shape[0], 1), -open_chain_joint_pos[:, 2])
        adjoint_theta3 = adjoint_transform(T_theta3)
        T_theta2 = exp_motion(self.thigh_axis_b.unsqueeze(0).repeat(open_chain_joint_pos.shape[0], 1), -open_chain_joint_pos[:, 1])
        T_theta3_mul_T_theta2 = torch.bmm(T_theta3, T_theta2)
        adjoint_theta2_theta3 = adjoint_transform(T_theta3_mul_T_theta2)
        body_jacobian[:, :, 1] = torch.bmm(
            adjoint_theta3, 
            self.thigh_axis_b.repeat(open_chain_joint_pos.shape[0]).view(-1, 6, 1)
        ).squeeze(2)
        body_jacobian[:, :, 0] = torch.bmm(
            adjoint_theta2_theta3,
            self.hip_axis_b.repeat(open_chain_joint_pos.shape[0]).view(-1, 6, 1)
        ).squeeze(2)
        if translation_only:
            body_jacobian = body_jacobian[:, 3:, :]  # Return only the translation part
        return body_jacobian
    
    def analytic_jacobian_open_chain(self, open_chain_joint_pos: torch.Tensor) -> torch.Tensor:
        """Calculate the analytic Jacobian for the given open chain joint positions.
        
        Args:
            open_chain_joint_pos (torch.Tensor): Open chain joint positions of shape (N, 3).
        """
        body_jacobian = self.body_jacobian_open_chain(open_chain_joint_pos, translation_only=True)
        Rsb = self.forward_kinematics_open_chain_full(open_chain_joint_pos)[:, :3, :3]
        analytic_jacobian = Rsb @ body_jacobian
        return analytic_jacobian

    '''Velocity Kinematics Functions'''
    def forward_velocity_kinematics(self, actuator_joint_pos: torch.Tensor, actuator_joint_vel: torch.Tensor) -> torch.Tensor:
        actuator_joint_pos = actuator_joint_pos.to(dtype=self.dtype)
        actuator_joint_vel = actuator_joint_vel.to(dtype=self.dtype)
        actuator_jacobian = self.actuator_jacobian(actuator_joint_pos)
        # Calculate the end-effector velocity
        end_effector_vel_b = torch.bmm(actuator_jacobian, actuator_joint_vel.unsqueeze(2)).squeeze(2)
        return end_effector_vel_b.to(dtype=self.output_dtype)
    
    def inverse_velocity_kinematics(self, goal_foot_vel: torch.Tensor, actuator_joint_pos: torch.Tensor) -> torch.Tensor:
        actuator_joint_pos = actuator_joint_pos.to(dtype=self.dtype)
        goal_foot_vel = goal_foot_vel.to(dtype=self.dtype)
        inv_actuator_jacobian = self.actuator_jacobian_inverse(actuator_joint_pos)
        # Calculate the actuator joint velocity
        actuator_joint_vel = torch.bmm(inv_actuator_jacobian, goal_foot_vel.unsqueeze(2)).squeeze(2)
        return actuator_joint_vel.to(dtype=self.output_dtype)

    '''Statics Functions'''
    def forward_statics(self, actuator_joint_pos: torch.Tensor, actuator_joint_torque: torch.Tensor) -> torch.Tensor:
        actuator_joint_pos = actuator_joint_pos.to(dtype=self.dtype)
        actuator_joint_torque = actuator_joint_torque.to(dtype=self.dtype)
        actuator_jacobian_inverse = self.actuator_jacobian_inverse(actuator_joint_pos)
        # Calculate the end-effector force
        end_effector_force = torch.bmm(actuator_jacobian_inverse.transpose(1, 2), actuator_joint_torque.unsqueeze(2)).squeeze(2)
        return end_effector_force.to(dtype=self.output_dtype)

    def inverse_statics(self, actuator_joint_pos: torch.Tensor, end_effector_force: torch.Tensor) -> torch.Tensor:
        actuator_joint_pos = actuator_joint_pos.to(dtype=self.dtype)
        end_effector_force = end_effector_force.to(dtype=self.dtype)
        actuator_jacobian = self.actuator_jacobian(actuator_joint_pos)
        # Calculate the actuator joint torque
        actuator_torque = torch.bmm(actuator_jacobian.transpose(1, 2), end_effector_force.unsqueeze(2)).squeeze(2)
        return actuator_torque.to(dtype=self.output_dtype)

class GeneralSphericalKinSolver(GeneralBennettKinSolver):
    def __init__(
            self, 
            h, x0, y0, z0, alpha,
            thigh_offset=0.0, knee_offset=0.0, model_knee_offset=0.0,
            device="cuda", 
            dtype=torch.float64, 
            output_dtype=torch.float32
        ):
        self.device = device
        self.dtype = dtype
        self.output_dtype = output_dtype

        # Constant mechanism parameters
        self.alpha = alpha
        self.h = h
        self.thigh_offset = thigh_offset
        self.knee_offset = knee_offset
        self.model_knee_offset = model_knee_offset
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

        # Initial transformation matrix
        self._M0 = torch.eye(4, device=self.device, dtype=self.dtype)
        self._M0[0, 3] = self.x0
        self._M0[1, 3] = self.y0
        self._M0[2, 3] = self.z0

        # Screw axes
        self.hip_axis = torch.tensor([1.0, 0.0, 0.0, 0.0, self.h, 0.0], device=device, dtype=dtype)  # Hip
        self.thigh_axis = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)  # Thigh
        self.knee_axis = torch.tensor([sin(self.alpha), 0.0, cos(self.alpha), 0.0, 0.0, 0.0], device=device, dtype=dtype)  # Knee                                          

        # Calculate body screw axis
        self._M0_inv = torch.inverse(self._M0)
        self.hip_axis_b = adjoint_transform(self._M0_inv.unsqueeze(0)).squeeze(0) @ self.hip_axis
        self.thigh_axis_b = adjoint_transform(self._M0_inv.unsqueeze(0)).squeeze(0) @ self.thigh_axis
        self.knee_axis_b = adjoint_transform(self._M0_inv.unsqueeze(0)).squeeze(0) @ self.knee_axis

        # Constants for analytical ik
        self.A = self.x0*cos(self.alpha) - self.z0*sin(self.alpha)
        self.D = self.h - self.z0
        self.E = 2 * ( self.x0*cos(self.alpha) + self.D*sin(self.alpha) )
        self.K = 2*self.A**2 + self.x0**2 + self.y0**2 + self.D**2 - self.A*self.E
        self.M = -2*self.A**2 + self.A*self.E
        self.N = 2*self.A*self.y0 - self.y0*self.E
        self.H = sqrt(self.M**2 + self.N**2)
        self.cos_alpha = cos(self.alpha)
        self.sin_alpha = sin(self.alpha)

    '''Open Chain - Actuator Model Mapping Functions'''
    def actuator_pos_from_open_chain(self, open_chain_joint_pos: torch.Tensor) -> torch.Tensor:
        actuator_pos = open_chain_joint_pos.clone()
        actuator_pos[:, 1] += self.thigh_offset
        # For calf joint, use spherical trigonometry to calculate the position
        theory_knee_pos = -open_chain_joint_pos[:, 2] + self.model_knee_offset
        X1 = self.sin_alpha ** 2 * torch.sin(theory_knee_pos)
        X2 = self.sin_alpha ** 2 * self.cos_alpha * (1 + torch.cos(theory_knee_pos))
        X3 = self.cos_alpha ** 3 - self.sin_alpha ** 2 * self.cos_alpha * torch.cos(theory_knee_pos) - self.cos_alpha
        theory_calf_pos = 2*torch.atan( (X1 + torch.sqrt(X1**2 + X2**2 - X3**2)) / (X2 - X3))
        model_calf = theory_calf_pos + self.knee_offset
        
        actuator_pos[:, 2] = -model_calf
        return actuator_pos
    
    def open_chain_pos_from_actuator(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        open_chain_pos = actuator_pos.clone()
        open_chain_pos[:, 1] -= self.thigh_offset
        # For knee joint, using spherical trigonometry to calculate the position
        theory_calf_pos = -actuator_pos[:, 2] - self.knee_offset
        alpha13 = torch.acos( self.cos_alpha**2 + self.sin_alpha**2 * torch.cos(theory_calf_pos) )
        gamma = torch.asin( torch.sin(theory_calf_pos) * self.sin_alpha / torch.sin(alpha13) )
        theory_knee = pi - 2*gamma
        model_knee = theory_knee - self.model_knee_offset
        open_chain_pos[:, 2] = -model_knee # Minus sign to keep the same direction as in the open chain model
        return open_chain_pos

    def knee_pos_from_actuator(self, actuator_pos: torch.Tensor) -> torch.Tensor:
        theory_calf_pos = -actuator_pos[:, 2] - self.knee_offset
        alpha13 = torch.acos( self.cos_alpha**2 + self.sin_alpha**2 * torch.cos(theory_calf_pos) )
        gamma = torch.asin( torch.sin(theory_calf_pos) * self.sin_alpha / torch.sin(alpha13) )
        theory_knee = pi - 2*gamma
        model_knee = theory_knee - self.model_knee_offset
        knee_pos = -model_knee
        return knee_pos

    def velocity_mapping(self, actuator_joint_pos: torch.Tensor) -> torch.Tensor:
        velocity_mapping = torch.eye(3, 3, device=self.device, dtype=self.dtype).repeat(actuator_joint_pos.shape[0], 1, 1)
        
        theory_calf_pos = -actuator_joint_pos[:, 2] - self.knee_offset

        g = self.cos_alpha**2 + self.sin_alpha**2 * torch.cos(theory_calf_pos)
        h = torch.sin(theory_calf_pos) * self.sin_alpha / torch.sin(torch.acos(g))

        d_gamma_d_h = 1.0 / torch.sqrt(1 - h**2)
        d_h_d_theta = -torch.cos(theory_calf_pos) * self.sin_alpha / torch.sin(torch.acos(g)) \
            + torch.sin(theory_calf_pos) * (self.sin_alpha/torch.sin(torch.acos(g))**2) * (g/torch.sqrt(1-g**2)) * self.sin_alpha**2 * torch.sin(theory_calf_pos)
        velocity_mapping[:, 2, 2] = 2 * d_gamma_d_h * d_h_d_theta

        return velocity_mapping

    def forward_kinematics_open_chain_full(self, span_tree_joint_pos: torch.Tensor) -> torch.Tensor:
        T = super().forward_kinematics_open_chain_full(span_tree_joint_pos)
        # Offset the z-axis translation by h
        T[:, 2, 3] -= self.h
        return T
    
    def inverse_kinematics_open_chain(
        self, pos_tensor: torch.Tensor, ref_pos_tensor: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = pos_tensor.shape[0]
        # There are totally 8 possible solutions
        ik_solutions = torch.zeros((batch_size, 8, 3), device=self.device, dtype=self.dtype)

        x = pos_tensor[:, 0:1]
        y = pos_tensor[:, 1:2]
        z = pos_tensor[:, 2:3]

        R = x**2 + y**2 + z**2
        S = R - self.K

        # 1. Check if the goal position is out of workspace in the radial/q3 dimension (Delta_q3 < 0)
        raw_mns_square_sum = self.M**2 + self.N**2 - S**2
        out_of_workspace_q3 = raw_mns_square_sum < -1e-6

        # Clamp S to [-H, H] and mns_square_sum to >= 0 so out-of-workspace targets project onto the boundary
        S_clamped = torch.clamp(S, min=-self.H, max=self.H)
        mns_square_sum = torch.clamp(self.M**2 + self.N**2 - S_clamped**2, min=0.0)
        sqrt_mns = torch.sqrt(mns_square_sum)

        # Avoid catastrophic cancellation regardless of the sign of self.N
        if self.N >= 0:
            q3_1 = 2 * torch.atan2(self.N + sqrt_mns, self.M + S_clamped)
            q3_2 = 2 * torch.atan2(S_clamped - self.M, self.N + sqrt_mns)
        else:
            q3_1 = 2 * torch.atan2(S_clamped - self.M, self.N - sqrt_mns)
            q3_2 = 2 * torch.atan2(self.N - sqrt_mns, self.M + S_clamped)

        ik_solutions[:, 0:4, 2] = q3_1.repeat(1, 4)
        ik_solutions[:, 4:8, 2] = q3_2.repeat(1, 4)

        out_of_workspace_branches = torch.zeros((batch_size, 8), device=self.device, dtype=torch.bool)

        for offset, q3 in [(0, q3_1), (4, q3_2)]:
            # Solve for q2 and q1
            P = (
                self.cos_alpha * self.A * torch.cos(q3)
                - self.cos_alpha * self.A + self.x0
                - self.y0 * self.cos_alpha * torch.sin(q3)
            )
            Q = -self.A * torch.sin(q3) - self.y0 * torch.cos(q3)
            U = (
                self.D - self.sin_alpha * self.A
                + self.sin_alpha * self.A * torch.cos(q3)
                - self.y0 * self.sin_alpha * torch.sin(q3)
            )
            square_sum = y**2 + z**2 - U**2
            out_of_workspace_q12 = (square_sum < -1e-6).squeeze(1)
            branch_out_of_ws = torch.logical_or(out_of_workspace_q3.squeeze(1), out_of_workspace_q12)
            out_of_workspace_branches[:, offset:offset + 4] = branch_out_of_ws.unsqueeze(1).repeat(1, 4)

            T = torch.sqrt(torch.clamp(square_sum, min=0.0))

            q2_1 = torch.atan2(Q * x - P * T, P * x + Q * T)
            q2_2 = torch.atan2(Q * x + P * T, P * x - Q * T)
            q1_1 = torch.atan2(U * y + T * z, T * y - U * z)
            q1_2 = torch.atan2(U * y - T * z, -T * y - U * z)

            pairs = [(q1_1, q2_1), (q1_1, q2_2), (q1_2, q2_1), (q1_2, q2_2)]
            for i, (q1_cand, q2_cand) in enumerate(pairs):
                ik_solutions[:, offset + i, 0:1] = q1_cand
                ik_solutions[:, offset + i, 1:2] = q2_cand

        # 4. Evaluate FK error on raw candidates
        fk_pos_tensor = self.forward_kinematics_open_chain(ik_solutions.view(-1, 3)).view(batch_size, 8, 3)
        ik_errors = torch.norm(fk_pos_tensor - pos_tensor.unsqueeze(1), dim=2)
        
        # 5. Select the best candidate
        if ref_pos_tensor is not None:
            ref_pos = ref_pos_tensor.to(dtype=self.dtype)
            min_ik_error, _ = torch.min(ik_errors, dim=1, keepdim=True)
            valid_fk_mask = ik_errors <= (min_ik_error + 1e-4)

            ang_diff = self.angle_diff(ik_solutions, ref_pos.unsqueeze(1))
            ref_pos_error = torch.norm(ang_diff, dim=2)

            masked_ref_pos_error = torch.where(
                valid_fk_mask,
                ref_pos_error,
                torch.full_like(ref_pos_error, float("inf"))
            )
            best_indices = torch.argmin(masked_ref_pos_error, dim=1)
        else:
            # When no ref_pos is provided, prefer physical knee solutions where sin(q3) <= 0 (i.e. q3 in [-pi, 0])
            within_q3_limit = torch.sin(ik_solutions[:, :, 2]) <= 1e-5
            penalized_ik_errors = torch.where(within_q3_limit, ik_errors, ik_errors + 100.0)
            best_indices = torch.argmin(penalized_ik_errors, dim=1)

        # Select the solution with the minimum error
        batch_indices = torch.arange(batch_size, device=self.device)
        best_solutions = ik_solutions[batch_indices, best_indices]
        best_errors = ik_errors[batch_indices, best_indices]
        out_of_workspace = torch.logical_or(
            out_of_workspace_branches[batch_indices, best_indices],
            best_errors > 1e-4
        )

        # 6. Convert ONLY the winning solution to the desired ranges at the very end
        best_solutions = self.format_joint_ranges(best_solutions)

        return best_solutions.to(dtype=self.output_dtype), best_errors.to(dtype=self.output_dtype), out_of_workspace
