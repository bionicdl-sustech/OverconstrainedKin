from ov_data import OvRobotDataV2, grid_plot
import os

data_paths = [
    'planar_y.pkl'
]
data_paths = [os.path.join(os.path.dirname(__file__), data_path) for data_path in data_paths]
nick_names = [
    "planar",
]
intervel_index = [250, 1000] # Example interval index for calculations

if __name__ == "__main__":
    ov_robot_data: dict[str, OvRobotDataV2] = {}
    for data_path, nick_name in zip(data_paths, nick_names):
        print(f"Loading data from {data_path} with nickname {nick_name}")
        ov_robot_data[nick_name] = OvRobotDataV2.from_pkl(data_path)

    grid_plot(
        ov_data_list=list(ov_robot_data.values()),
        data_name="joint_positions",
        nick_names=nick_names,
        num_rows=3,
        num_cols=2,
        interval_index=intervel_index,
        offset_time=True,
        share_x=True,
        share_y=False,
        title=f"Joint Positions",
        transpose_row_col=True
    )

    grid_plot(
        ov_data_list=list(ov_robot_data.values()),
        data_name="joint_velocities",
        nick_names=nick_names,
        num_rows=3,
        num_cols=2,
        interval_index=intervel_index,
        offset_time=True,
        share_x=True,
        share_y=True,
        title=f"Joint Velocities",
        transpose_row_col=True
    )

    grid_plot(
        ov_data_list=list(ov_robot_data.values()),
        data_name="feet_positions_b",
        nick_names=nick_names,
        num_rows=2,
        num_cols=3,
        interval_index=intervel_index,
        offset_time=True,
        share_x=True,
        share_y=True,
        title=f"Feet Positions in Body Frame",
        transpose_row_col=False
    )

    grid_plot(
        ov_data_list=list(ov_robot_data.values()),
        data_name="feet_velocities_b",
        nick_names=nick_names,
        num_rows=2,
        num_cols=3,
        interval_index=intervel_index,
        offset_time=True,
        share_x=True,
        share_y=True,
        title=f"Feet Velocities in Body Frame",
        transpose_row_col=False
    )

    import matplotlib.pyplot as plt
    plt.show()