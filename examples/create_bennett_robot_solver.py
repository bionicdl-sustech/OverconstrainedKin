from ov_kin.ov_robot_kinematics import OvRobotKinSolver, GeneralBennettKinCfg, RobotCfg
from math import pi
import torch

custom_config = GeneralBennettKinCfg(
    l=0.162,
    x0=0.377,
    y0=0.018,
    z0=0.07,
    zt=0.0746188,
    beta=60 * pi / 180.0,
    alpha=120 * pi / 180.0,
)

custom_robot_cfg = RobotCfg(
    length=0.38,
    width=0.096,
    actuator_signs_list=[
        [1, 1, 1],  # FL
    ],
    foot_pos_signs_list=[
        [1, 1, 1],  # FL
    ],
)

custom_robot_solver = OvRobotKinSolver(
    kin_cfg=custom_config,
    robot_cfg=custom_robot_cfg,
    device="cpu",
    dtype=torch.float64,
    output_dtype=torch.float32,
)

hip_pos_range = [-pi, pi]
thigh_pos_range = [0, 1.5 * pi]
calf_pos_range = [-170 / 180 * pi, 0]

# Sample random actuator positions within the specified ranges
num_samples = 16384
hip_pos_samples = torch.empty(num_samples).uniform_(*hip_pos_range)
thigh_pos_samples = torch.empty(num_samples).uniform_(*thigh_pos_range)
calf_pos_samples = torch.empty(num_samples).uniform_(*calf_pos_range)
actuator_pos_samples = torch.stack([hip_pos_samples, thigh_pos_samples, calf_pos_samples], dim=1).unsqueeze(1)
fk_result = custom_robot_solver.fk(actuator_pos_samples)
print("Forward Kinematics Result:", fk_result[0])

# Add some noise to the actuator positions as reference
noise_level = 0.0  # Adjust the noise level as needed
ref_actuator_pos_samples = actuator_pos_samples + torch.randn_like(actuator_pos_samples) * noise_level
ref_actuator_pos_samples[:, :, 0] = torch.clip(ref_actuator_pos_samples[:, :, 0], min=hip_pos_range[0], max=hip_pos_range[1])
ref_actuator_pos_samples[:, :, 1] = torch.clip(ref_actuator_pos_samples[:, :, 1], min=thigh_pos_range[0], max=thigh_pos_range[1])
ref_actuator_pos_samples[:, :, 2] = torch.clip(ref_actuator_pos_samples[:, :, 2], min=calf_pos_range[0], max=calf_pos_range[1])

# Inverse Kinematics Test
foot_pos_tensor = fk_result.clone()
actuator_pos_result, ik_errors, out_of_workspace = custom_robot_solver.ik(foot_pos_tensor, ref_actuator_pos_samples)

# Count the number of successful IK solutions (where the error is below a threshold)
success_threshold = 1e-3
successful_ik_count = (ik_errors < success_threshold).sum().item()
print(f"Number of successful IK solutions: {successful_ik_count} out of {num_samples}")
print(f"Number of out of workspace solutions: {out_of_workspace.sum().item()} out of {num_samples}")

# Print the maximum IK error and the corresponding actuator positions for the worst case
max_ik_error = ik_errors.max().item()
worst_case_index = ik_errors.argmax().item()
print(f"Maximum IK error: {max_ik_error}")
print(f"Actuator positions for worst case: {actuator_pos_samples[worst_case_index].squeeze()}")

# Check if the ik result fit the original actuator positions
joint_space_errors = torch.norm(actuator_pos_result - actuator_pos_samples, dim=-1).squeeze()
joint_space_success_threshold = 1e-2
joint_space_success_count = (joint_space_errors < joint_space_success_threshold).sum().item()
print(f"Number of successful IK solutions in joint space: {joint_space_success_count} out of {num_samples}")
joint_space_worst_case_index = joint_space_errors.argmax().item()
print(f"Worst case solved actuator pos: {actuator_pos_result[joint_space_worst_case_index].squeeze()}")
print(f"Worst case ground truth actuator pos: {actuator_pos_samples[joint_space_worst_case_index]}")

# IK again to find possible issues
worst_case_foot_pos = fk_result[joint_space_worst_case_index].unsqueeze(1)
worst_case_actuator_ref_pos = ref_actuator_pos_samples[joint_space_worst_case_index].unsqueeze(1)
worst_case_ik_result, _, _ = custom_robot_solver.ik(worst_case_foot_pos, worst_case_actuator_ref_pos)
print("Worst case ik result:", worst_case_ik_result)
