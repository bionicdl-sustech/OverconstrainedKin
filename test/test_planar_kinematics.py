import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ov_kin import MagicLimbPlanarKin
import numpy as np

if __name__ == "__main__":
    planar_kin = MagicLimbPlanarKin()

    # Test forward kinematics
    zero_actuator_joint_pos = np.array([0, 0, 0])
    actuator_joint_pos = np.array([-np.pi/3, 2*np.pi/4, -np.pi/3])
    pos, T = planar_kin.forward_kinematics(actuator_joint_pos)
    print(f"Forward Kinematics Position (MR): {pos}")
    pos, T = planar_kin.forward_kinematics_rtb(actuator_joint_pos)
    print(f"Forward Kinematics Position (RTB): {pos}")

    # Verify inverse kinematics
    pos = [ 0.0350,  0.2606, -0.2800]
    ik_actuator_pos, error = planar_kin.inverse_kinematics(goal_pos_in_hip_frame=pos, ik_func="ik_GN")
    ik_actuator_pos_exp, analytic_error = planar_kin.analytic_ik(goal_pos_in_hip_frame=pos)
    print(f"Numerical Inverse Kinematics Actuator Position: {ik_actuator_pos}")
    print(f"Analytic Inverse Kinematics Actuator Position: {ik_actuator_pos_exp}")
    print(f"Actuator Joint Positions: {actuator_joint_pos}")

    # Verify analytic jacobian
    jacobian = planar_kin.analytic_linear_jacobian(actuator_joint_pos)
    print(f"Analytic Jacobian (MR):\n{jacobian}")

    jacobian = planar_kin.analytic_linear_jacobian_rtb(actuator_joint_pos)
    print(f"Analytic Jacobian (RTB):\n{jacobian}")

    # Verify forward velocity kinematics
    forward_velocity = planar_kin.forward_velocity_kinematics(zero_actuator_joint_pos, np.array([0.0, 0.0, 1.0]))
    print(f"Forward Velocity Kinematics (MR): {forward_velocity}")
    
    forward_velocity = planar_kin.forward_velocity_kinematics_rtb(zero_actuator_joint_pos, np.array([0.0, 0.0, 1.0]))
    print(f"Forward Velocity Kinematics (RTB): {forward_velocity}")

    actuator_joint_vel = planar_kin.inverse_velocity_kinematics(goal_vel_in_hip_frame=[-0.0, 0.3, 0.0], actuator_joint_pos=zero_actuator_joint_pos)
    print(f"Inverse Velocity Kinematics Actuator Joint Velocities: {actuator_joint_vel}")

    # Visualize
    planar_kin.visualize(ik_actuator_pos, backend="swift", block=True)