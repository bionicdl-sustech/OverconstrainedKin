from ov_kin.ov_robot_kinematics import (
    OvRobotKinSolver, 
    RobotCfg, 
    GeneralBennettKinCfg, 
    GeneralSphericalKinCfg
)

from math import pi, atan
import torch

as_over_dog_open_bennett_kin_cfg = GeneralBennettKinCfg(
    l = 0.162,

    x0 = 0.377,
    y0 = 0.018,
    z0 = 0.07,
    zt = 0.0746188,

    beta = 60 * pi / 180.0,
    alpha = 120 * pi / 180.0
)

as_over_dog_open_planar_kin_cfg = GeneralBennettKinCfg(
    l = 0.16252077,
    
    x0 = 0.37643345,
    y0 = 0.02216096,
    z0 = 0.07,
    zt = 0.07,

    beta = 0,
    alpha = pi,

    thigh_offset = -atan(0.013/0.162),
    knee_offset = atan(0.013/0.162)*2
)

as_over_dog_open_spherical_kin_cfg = GeneralSphericalKinCfg(
    h = 0.23,
    
    x0 = 0.29007149,
    y0 = -0.00565786,
    z0 = 0.09465475,

    alpha = 31.02397 * pi / 180.0,

    thigh_offset = -4.67444*pi/180.0,
    knee_offset = -9.34888 * pi / 180.0,
    model_knee_offset = 16.03251 / 2 * pi / 180.0
)

as_over_dog_open_robot_cfg = RobotCfg(
    length = 0.38,
    width = 0.096,
    actuator_signs_list = [
        [1, 1, 1], # FL
        [-1, -1, -1], # FR
        [-1, -1, -1], # RL
        [1, 1, 1], # RR 
    ],
    foot_pos_signs_list = [
        [1, 1, 1], # FL
        [1, -1, 1], # FR
        [-1, 1, 1], # RL
        [-1, -1, 1], # RR 
    ]
)

as_over_dog_open_fl_cfg = RobotCfg(
    length = 0.38,
    width = 0.096,
    actuator_signs_list = [
        [1, 1, 1], # FL only 
    ],
    foot_pos_signs_list = [
        [1, 1, 1], # FL only
    ]
)
    

class AsOvDogOpenBennettSolver(OvRobotKinSolver):
    def __init__(
        self,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        super().__init__(
            kin_cfg=as_over_dog_open_bennett_kin_cfg,
            robot_cfg=as_over_dog_open_robot_cfg,
            device=device, 
            dtype=dtype, 
            output_dtype=output_dtype
        )

class AsOvDogOpenBennettFLSolver(OvRobotKinSolver):
    '''
        Single-limb (front left leg) kinematics solver for the AsOvDogOpenBennett
    '''
    def __init__(
        self,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        super().__init__(
            kin_cfg=as_over_dog_open_bennett_kin_cfg,
            robot_cfg=as_over_dog_open_fl_cfg,
            device=device, 
            dtype=dtype, 
            output_dtype=output_dtype
        )

class AsOvDogOpenPlanarSolver(OvRobotKinSolver):
    def __init__(
        self,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        super().__init__(
            kin_cfg=as_over_dog_open_planar_kin_cfg,
            robot_cfg=as_over_dog_open_robot_cfg,
            device=device, 
            dtype=dtype, 
            output_dtype=output_dtype
        )

class AsOvDogOpenPlanarFLSolver(OvRobotKinSolver):
    '''
        Single-limb (front left leg) kinematics solver for the AsOvDogOpenBennett
    '''
    def __init__(
        self,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        super().__init__(
            kin_cfg=as_over_dog_open_planar_kin_cfg,
            robot_cfg=as_over_dog_open_fl_cfg,
            device=device, 
            dtype=dtype, 
            output_dtype=output_dtype
        )


class AsOvDogOpenSphericalSolver(OvRobotKinSolver):
    def __init__(
        self,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        super().__init__(
            kin_cfg=as_over_dog_open_spherical_kin_cfg,
            robot_cfg=as_over_dog_open_robot_cfg,
            device=device, 
            dtype=dtype, 
            output_dtype=output_dtype
        )

class AsOvDogOpenSphericalFLSolver(OvRobotKinSolver):
    '''
        Single-limb (front left leg) kinematics solver for the AsOvDogOpenBennett
    '''
    def __init__(
        self,
        device="cuda", 
        dtype=torch.float64, 
        output_dtype=torch.float32
    ):
        super().__init__(
            kin_cfg=as_over_dog_open_spherical_kin_cfg,
            robot_cfg=as_over_dog_open_fl_cfg,
            device=device, 
            dtype=dtype, 
            output_dtype=output_dtype
        )