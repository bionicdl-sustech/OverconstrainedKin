# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""
import os
import argparse
from tqdm import tqdm
from isaaclab.app import AppLauncher
from ov_data import OvRobotDataV2

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--decimation", type=int, default=2, help="Number of decimation steps to take before updating joint pos target.")
parser.add_argument("--explicit_ik", default=True,  action="store_true")
parser.add_argument("--save_path", type=str, default="planar_y")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import Articulation
from isaaclab.utils import configclass
import numpy as np
import json

from overconstrained_utils.kinematics_solver import RTBKinematicsSolver
from overconstrained_utils.foot_trajectory_generator import PhaseGuidedTrajectoryGenerator
from ov_kin import PlanarKinematics

##
# Pre-defined configs
##
from overconstrained_assets.magic_limb.magic_limb_robot_cfg import (
    MagicLimbRobotPlanarFL,
    MagicLimbRobotBennettFL,
)
from overconstrained_assets import ASSET_ROOT

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot = MagicLimbRobotPlanarFL.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    # Move the robot to a high position
    robot.init_state.pos = (0.0, 0.0, 0.0)
    # Fix root
    robot.spawn.articulation_props.fix_root_link = True

    # robot.actuators["motors"].stiffness = 30.0
    # robot.actuators["motors"].damping = 0.7

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    ik_solver = PlanarKinematics()
    traj_gen = PhaseGuidedTrajectoryGenerator(dt=sim.get_physics_dt()*args_cli.decimation, foot_radius=0.02)
    traj_gen.set_gait_parameters(
        velocity=[0.0, 1.25],
        gait_freq=2.0,
        base_height=0.3,
        stance_width=0.3,
        stance_length=0.45,
        foot_clearance=0.04,
        stance_duty_factor=0.5
    )

    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot: Articulation = scene["robot"]

    # Markers
    feasible_marker_cfg = CONTACT_SENSOR_MARKER_CFG.copy()
    feasible_marker_cfg.markers["contact"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
    feasible_marker_cfg.markers["contact"].radius = 0.03
    feasible_marker_cfg = VisualizationMarkers(feasible_marker_cfg.replace(prim_path="/Visuals/goal"))
    infeasible_marker_cfg = CONTACT_SENSOR_MARKER_CFG.copy()
    infeasible_marker_cfg.markers["contact"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
    infeasible_marker_cfg.markers["contact"].radius = 0.03
    infeasible_marker_cfg = VisualizationMarkers(infeasible_marker_cfg.replace(prim_path="/Visuals/goal"))

    # Specify robot-specific parameters
    robot_entity_cfg_FL = SceneEntityCfg("robot", joint_names=["motor_hip_FL", "motor_thigh_FL", "motor_calf_FL",], body_names=["FootTip_FL"])
    actuated_joints_cfg = SceneEntityCfg("robot", joint_names="motor.*")
    # Resolving the scene entities
    robot_entity_cfg_FL.resolve(scene)
    actuated_joints_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_reset = False
    sim_count = 0

    # Simulation loop
    ov_robot_data = OvRobotDataV2()
    ik_errors = []
    for _ in tqdm(range(500*args_cli.decimation)):
        if not sim_reset:
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            sim_reset = True
            continue

        # Check position/velocity relationship between knee and calf
        if sim_count % args_cli.decimation == 0:
            pos_FL, vel_FL = traj_gen.generate_foot_trajectory_with_vel()

            # Get inverse kinematics
            ik_pos_FL = [pos_FL[0]-0.19, pos_FL[1]-0.048, pos_FL[2]]
            if args_cli.explicit_ik:
                # Offset the position
                sol_joint_pos_FL, error = ik_solver.explicit_ik(ik_pos_FL)
                ik_errors.append(error)
                if error > 0.01:
                    success_FL = False
                else:
                    success_FL = True
            else:
                sol_joint_pos_FL, success_FL = ik_solver.inverse_kinematics(
                    goal_pos_in_hip_frame=ik_pos_FL,
                    ik_func="ik_GN"
                )
                ik_errors.append(0.0 if success_FL else 1.0)

            sol_joint_vel_FL = ik_solver.inverse_velocity_kinematics(
                goal_vel_in_hip_frame=vel_FL,
                actuator_joint_pos=sol_joint_pos_FL
            )
            goal_joint_pos_tensor_FL = torch.zeros((1, 3), device=sim.device)
            goal_joint_pos_tensor_FL[:, 0] = sol_joint_pos_FL[0]
            goal_joint_pos_tensor_FL[:, 1] = sol_joint_pos_FL[1]
            goal_joint_pos_tensor_FL[:, 2] = sol_joint_pos_FL[2]

            # update marker positions
            goal_pos_tensor = torch.tensor([pos_FL], device=sim.device)
            if success_FL:
                feasible_marker_cfg.visualize(goal_pos_tensor + scene.env_origins.repeat(4, 1))
            else:
                infeasible_marker_cfg.visualize(goal_pos_tensor + scene.env_origins.repeat(4, 1))

        # for FL control
        robot.set_joint_position_target(goal_joint_pos_tensor_FL, joint_ids=robot_entity_cfg_FL.joint_ids)
        
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update buffers
        scene.update(sim_dt)
        
        timestamp = sim_dt * sim_count
        joint_names = [robot.data.joint_names[i] for i in actuated_joints_cfg.joint_ids]
        joint_names_full = [robot.data.joint_names[i] for i in actuated_joints_cfg.joint_ids]
        joint_names_full.extend(["ik_motor_hip_FL", "ik_motor_thigh_FL", "ik_motor_calf_FL"])
        joint_pos = robot.data.joint_pos[0, actuated_joints_cfg.joint_ids].cpu().numpy().tolist()
        joint_pos_full = np.zeros(6, dtype=np.float32)
        joint_pos_full[:3] = joint_pos
        joint_pos_full[3:] = sol_joint_pos_FL
        joint_vel = robot.data.joint_vel[0, actuated_joints_cfg.joint_ids].cpu().numpy().tolist()
        joint_vel_full = np.zeros(6, dtype=np.float32)
        joint_vel_full[:3] = joint_vel
        joint_vel_full[3:] = sol_joint_vel_FL
        feet_FL_pos_b = robot.data.body_pos_w[0, robot_entity_cfg_FL.body_ids].squeeze(0).cpu().numpy().tolist()
        feet_FL_vel_b = robot.data.body_lin_vel_w[0, robot_entity_cfg_FL.body_ids].squeeze(0).cpu().numpy().tolist()
        ov_robot_data.add_joint_positions(
            timestamp=timestamp,
            joint_positions=joint_pos_full,
            joint_names=joint_names_full
        )
        ov_robot_data.add_joint_velocities(
            timestamp=timestamp,
            joint_velocities=joint_vel_full,
            joint_names=joint_names_full
        )
        ov_robot_data.add_feet_positions_b(
            timestamp=timestamp,
            feet_positions_b=[
                feet_FL_pos_b,
                pos_FL
            ],
            feet_names=[
                "FootTip_FL",
                "GoalFootTip_FL"
            ]
        )
        ov_robot_data.add_feet_velocities_b(
            timestamp=timestamp,
            feet_velocities_b=[
                feet_FL_vel_b,
                vel_FL
            ],
            feet_names=[
                "FootTip_FL",
                "GoalFootTip_FL"
            ]
        )

        sim_count += 1
    
    # Save the data to a file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ov_robot_data_path = os.path.join(current_dir, f"data/{args_cli.save_path}.pkl")
    ov_robot_data.to_pkl(ov_robot_data_path)

    # Plot the ik errors
    import matplotlib.pyplot as plt
    print(f"Max IK Error: {max(ik_errors)}")
    plt.plot(ik_errors)
    plt.xlabel("Time Step")
    plt.ylabel("IK Error")
    plt.title("Inverse Kinematics Error Over Time")
    plt.grid()
    plt.show()

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.0025, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()