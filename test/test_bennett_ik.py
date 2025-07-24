NUM_CONFIGS = 100000
SUCCESS_THRESHOLD = 1e-3
SUCCESS_THRESHOLD_JOINT = 1e-2

from ov_kin import BennettKinematics
import numpy as np
import math

if __name__ == "__main__":
    # generate random joint configurations
    hip_pos = np.random.uniform(-np.pi, 0, (NUM_CONFIGS, 1))
    thigh_pos = np.random.uniform(0.0, np.pi, (NUM_CONFIGS, 1))
    calf_pos = np.random.uniform(-130*np.pi/180, 0, (NUM_CONFIGS, 1))

    actuator_joint_pos = np.concatenate(
        (hip_pos, thigh_pos, calf_pos), axis=1
    )

    # create BennettKinematics instance
    bennett_kin = BennettKinematics()

    ik_erros = []
    hip_pos_errors = []
    thigh_pos_errors = []
    calf_pos_errors = []
    failed_xyz = []
    domain_error_failed_xyz = []
    for i in range(NUM_CONFIGS):
        current_joint_pos = actuator_joint_pos[i]
        # forward kinematics
        pos, T = bennett_kin.forward_kinematics(current_joint_pos)
        # print(f"Forward Kinematics Position (MR): {pos}")

        # analytic inverse kinematics
        try:
            ik_actuator_pos_exp, analytic_error, ik_sol = bennett_kin.analytic_ik(goal_pos_in_hip_frame=pos)
        except Exception as e:
            print(f"Error in analytic IK for configuration {i}: {e}")
            failed_xyz.append(pos)
            domain_error_failed_xyz.append(pos)
            continue
        # print(f"analytic Inverse Kinematics Actuator Position: {ik_actuator_pos_exp}")

        # Calculate joint pos errors
        hip_pos_error = abs(current_joint_pos[0] - ik_actuator_pos_exp[0])
        thigh_pos_error = abs(current_joint_pos[1] - ik_actuator_pos_exp[1])
        calf_pos_error = abs(current_joint_pos[2] - ik_actuator_pos_exp[2])

        hip_pos_errors.append(hip_pos_error)
        thigh_pos_errors.append(thigh_pos_error)
        calf_pos_errors.append(calf_pos_error)

        ik_erros.append(analytic_error)

        if analytic_error > SUCCESS_THRESHOLD or \
            hip_pos_error > SUCCESS_THRESHOLD_JOINT or \
                thigh_pos_error > SUCCESS_THRESHOLD_JOINT or \
                    calf_pos_error > SUCCESS_THRESHOLD_JOINT:
            failed_xyz.append(pos)

    # Print the average errors
    print(f"Average IK Error: {np.mean(ik_erros)}")
    print(f"Average Hip Position Error: {np.mean(hip_pos_errors)}")
    print(f"Average Thigh Position Error: {np.mean(thigh_pos_errors)}")
    print(f"Average Calf Position Error: {np.mean(calf_pos_errors)}")

    # Print the maximum errors
    print(f"Max IK Error: {np.max(ik_erros)}")
    print(f"Max Hip Position Error: {np.max(hip_pos_errors)}")
    print(f"Max Thigh Position Error: {np.max(thigh_pos_errors)}")
    print(f"Max Calf Position Error: {np.max(calf_pos_errors)}")

    # Calculate success rates
    ik_success_rate = np.mean(np.array(ik_erros) < SUCCESS_THRESHOLD)
    hip_pos_success_rate = np.mean(np.array(hip_pos_errors) < SUCCESS_THRESHOLD_JOINT)
    thigh_pos_success_rate = np.mean(np.array(thigh_pos_errors) < SUCCESS_THRESHOLD_JOINT)
    calf_pos_success_rate = np.mean(np.array(calf_pos_errors) < SUCCESS_THRESHOLD_JOINT)

    print(f"IK Success Rate: {ik_success_rate * 100:.2f}%")
    print(f"Hip Position Success Rate: {hip_pos_success_rate * 100:.2f}%")
    print(f"Thigh Position Success Rate: {thigh_pos_success_rate * 100:.2f}%")
    print(f"Calf Position Success Rate: {calf_pos_success_rate * 100:.2f}%")
    print(f"Total Configurations: {NUM_CONFIGS}")
    print(f"Total Successful Configurations: {np.sum(np.array(ik_erros) < SUCCESS_THRESHOLD)}")
    print(f"Total Successful Hip Configurations: {np.sum(np.array(hip_pos_errors) < SUCCESS_THRESHOLD_JOINT)}")
    print(f"Total Successful Thigh Configurations: {np.sum(np.array(thigh_pos_errors) < SUCCESS_THRESHOLD_JOINT)}")
    print(f"Total Successful Calf Configurations: {np.sum(np.array(calf_pos_errors) < SUCCESS_THRESHOLD_JOINT)}")

    if failed_xyz:
        print(f"Failed XYZ Positions: {len(failed_xyz)}")
        # Get the average z position of failed configurations
        avg_failed_z = np.mean([pos[2] for pos in failed_xyz])
        print(f"Average Z Position of Failed Configurations: {avg_failed_z:.4f}")
        # Print min and max z positions of failed configurations
        min_failed_z = np.min([pos[2] for pos in failed_xyz])
        max_failed_z = np.max([pos[2] for pos in failed_xyz])
        print(f"Min Z Position of Failed Configurations: {min_failed_z:.4f}")
        print(f"Max Z Position of Failed Configurations: {max_failed_z:.4f}")

    if domain_error_failed_xyz:
        print(f"Domain Error Failed XYZ Positions: {len(domain_error_failed_xyz)}")
        # Get the average z position of domain error failed configurations
        avg_domain_error_failed_z = np.mean([pos[2] for pos in domain_error_failed_xyz])
        print(f"Average Z Position of Domain Error Failed Configurations: {avg_domain_error_failed_z:.4f}")
        # Print min and max z positions of domain error failed configurations
        min_domain_error_failed_z = np.min([pos[2] for pos in domain_error_failed_xyz])
        max_domain_error_failed_z = np.max([pos[2] for pos in domain_error_failed_xyz])
        print(f"Min Z Position of Domain Error Failed Configurations: {min_domain_error_failed_z:.4f}")
        print(f"Max Z Position of Domain Error Failed Configurations: {max_domain_error_failed_z:.4f}")
    else:
        print("No domain error failed configurations found.")