# OverconstrainedKin (`ov_kin`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17ucathft0o7IDp-F7B2NsSsWmq3Kh4eW?usp=sharing)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**`ov_kin`** is a batched, analytical PyTorch kinematics, velocity, and statics solver for quadruped robots with **overconstrained robotic limbs**—including **Spatial Bennett 4-bar**, **Planar 4-bar**, and **Spherical 4-bar** linkages.

By decoupling each closed-loop linkage into an equivalent open kinematic chain paired with an exact closed-form actuator-to-joint mapping, `ov_kin` achieves:
- **Zero-Iteration Analytical Inverse Kinematics (IK):** Evaluates all 8 geometric branches simultaneously in batched PyTorch tensors (CPU or CUDA) without numerical root-finding or GPU-CPU synchronization bottlenecks.
- **Graceful Out-of-Workspace Projection:** Automatically projects unreachable goal foot positions onto the closest reachable workspace boundary while returning an explicit `out_of_workspace` boolean mask and residual FK error.
- **Analytical Jacobians, Velocity Kinematics & Statics:** Closed-form body/spatial Jacobians, forward/inverse velocity kinematics, and forward/inverse statics (joint torque $\leftrightarrow$ foot force mapping).
- **Interactive 3D Web Visualizer (`Viser`):** Includes a browser-based 3D URDF playground (`app.py`) with interactive 3D foot target gizmos and joint sliders.

---

## Supported Limb Morphologies

The repository includes pre-configured single-limb (`FL`) and full-quadruped (`FL, FR, RL, RR`) solvers for the **AsOverDogOpen** platform across three linkage morphologies:

| Morphology | Full-Robot Solver Class | Single-Limb (`FL`) Class | Config Class |
| :--- | :--- | :--- | :--- |
| **Bennett 4-Bar Linkage** | `AsOvDogOpenBennettSolver` | `AsOvDogOpenBennettFLSolver` | `GeneralBennettKinCfg` |
| **Planar 4-Bar Linkage** | `AsOvDogOpenPlanarSolver` | `AsOvDogOpenPlanarFLSolver` | `GeneralBennettKinCfg` ($\beta=0, \alpha=\pi$) |
| **Spherical 4-Bar Linkage** | `AsOvDogOpenSphericalSolver` | `AsOvDogOpenSphericalFLSolver` | `GeneralSphericalKinCfg` |

---

## Repository Structure

```text
OverconstrainedKin/
├── ov_kin/                             # Core Python package
│   ├── __init__.py                     # Public API exports
│   ├── base_kinematics.py              # GeneralBennettKinSolver & GeneralSphericalKinSolver
│   ├── ov_robot_kinematics.py          # Multi-limb OvRobotKinSolver wrapper & dataclasses
│   └── as_over_dog_open_kinematics.py  # Pre-configured AsOverDog solvers (Bennett, Planar, Spherical)
├── examples/                           # Validation & usage examples
│   ├── as_over_dog_open_bennett_solver.py
│   ├── as_over_dog_open_planar_solver.py
│   ├── as_over_dog_open_spherical_solver.py
│   └── create_bennett_robot_solver.py  # Custom robot configuration example
├── assets/                             # URDF descriptions & mesh directories
│   ├── AsOverDogOpenBennett.urdf
│   ├── AsOverDogOpenPlanar.urdf
│   ├── AsOverDogOpenSpherical.urdf
│   ├── meshes/                         # Place visual .stl/.glb meshes here
│   └── collision_meshes/               # Place collision .stl meshes here
├── app.py                              # Interactive 3D web visualizer (Viser)
├── Dockerfile                          # Container spec for Hugging Face Spaces / cloud deployment
├── requirements.txt                    # Dependencies for solver + 3D web visualizer
└── pyproject.toml                      # Package build configuration
```

---

## Installation

### 1. Core Kinematics Solver Only
If you only need the PyTorch kinematics solver (e.g., inside an Isaac Lab / Isaac Gym reinforcement learning environment):

```bash
git clone https://github.com/bionicdl-sustech/OverconstrainedKin.git
cd OverconstrainedKin
pip install -e .
```

### 2. Solver + Interactive 3D Web Visualizer
To run the interactive 3D URDF visualizer (`app.py`):

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1. Using Pre-Configured Quadruped Solvers (`FL, FR, RL, RR`)

All multi-limb solvers accept batched PyTorch tensors of shape `(batch_size, num_limbs, 3)`:
- **Actuator order:** `[hip, thigh, calf]` in radians.
- **Foot position order:** `[x, y, z]` in meters (in the robot's `MainBody` frame).

```python
import torch
from ov_kin import (
    AsOvDogOpenBennettSolver,
    AsOvDogOpenPlanarSolver,
    AsOvDogOpenSphericalSolver,
)

# Initialize full 4-limb solver (supports device="cuda" or "cpu")
solver = AsOvDogOpenBennettSolver(
    device="cpu",
    dtype=torch.float64,
    output_dtype=torch.float32,
)

# Batch of 2 robots, 4 limbs [FL, FR, RL, RR], 3 joints [hip, thigh, calf]
actuator_pos = torch.tensor([
    [[-0.67,  1.91, -0.74], [ 0.67, -1.91,  0.74], [ 0.67, -1.91,  0.74], [-0.67,  1.91, -0.74]],
    [[-0.50,  1.70, -0.90], [ 0.50, -1.70,  0.90], [ 0.50, -1.70,  0.90], [-0.50,  1.70, -0.90]],
], dtype=torch.float64)

# 1. Forward Kinematics (Actuator Positions -> Foot Positions)
foot_pos = solver.fk(actuator_pos)  # Shape: (2, 4, 3)

# 2. Inverse Kinematics (Goal Foot Positions + Reference Actuator Pos -> Actuator Positions)
solved_act_pos, ik_errors, out_of_workspace = solver.ik(
    foot_pos=foot_pos.to(torch.float64),
    ref_actuator_pos=actuator_pos,
)
# solved_act_pos:   (2, 4, 3) solved actuator angles
# ik_errors:        (2, 4)    Euclidean FK residual error [m]
# out_of_workspace: (2, 4)    Boolean mask (True if goal position is outside the limb workspace)

# 3. Passive Knee Joint Angles for URDF Visualization
knee_pos = solver.knee_pos_from_actuator(actuator_pos, use_urdf_convention=True)

# 4. Analytical Jacobian, Velocity Kinematics & Statics
J = solver.J(actuator_pos)                 # Shape: (2, 4, 3, 3)
J_inv = solver.J_inverse(actuator_pos)     # Shape: (2, 4, 3, 3)

actuator_vel = torch.zeros_like(actuator_pos)
foot_vel = solver.forward_velocity_kinematics(actuator_pos, actuator_vel)
recovered_act_vel = solver.inverse_velocity_kinematics(actuator_pos, foot_vel.to(torch.float64))

actuator_torque = torch.ones_like(actuator_pos)
foot_force = solver.forward_statics(actuator_pos, actuator_torque)
recovered_torque = solver.inverse_statics(actuator_pos, foot_force.to(torch.float64))
```

### 2. Running the Validation Examples

```bash
python examples/as_over_dog_open_bennett_solver.py
python examples/as_over_dog_open_planar_solver.py
python examples/as_over_dog_open_spherical_solver.py
python examples/create_bennett_robot_solver.py
```

---

## Interactive 3D Web Visualizer

The repository includes an interactive 3D web application (`app.py`) powered by [Viser](https://viser.studio/). Users can switch between the **Bennett**, **Planar**, and **Spherical** robot morphologies, drag 3D XYZ transform gizmos on each foot to test real-time IK, or adjust individual actuator sliders (`q1, q2, q3`) to inspect FK and closed-loop linkage motion.

1. Place the visual `.stl` mesh files into `assets/meshes/` (and optional collision meshes into `assets/collision_meshes/`).
2. Start the visualizer server:
   ```bash
   python app.py
   ```
3. Open **`http://localhost:7860`** in your browser.

*(A `Dockerfile` is also provided for one-click cloud deployment on Hugging Face Spaces.)*

---

## Custom Limb Morphologies & Parameter Identification (Coming Soon)

You can define custom overconstrained limb geometries using `GeneralBennettKinCfg`, `GeneralSphericalKinCfg`, and `RobotCfg` (see [`examples/create_bennett_robot_solver.py`](examples/create_bennett_robot_solver.py)):

```python
from math import pi
from ov_kin import OvRobotKinSolver, GeneralBennettKinCfg, RobotCfg

kin_cfg = GeneralBennettKinCfg(
    l=0.162,           # Linkage length [m]
    x0=0.377,          # Initial foot x position [m]
    y0=0.018,          # Initial foot y position [m]
    z0=0.070,          # Initial foot z position [m]
    zt=0.0746188,      # Knee joint origin z offset [m]
    beta=60 * pi / 180.0,
    alpha=120 * pi / 180.0,
    thigh_offset=0.0,
    knee_offset=0.0,
)
```

> 📌 **Note on Kinematic Parameter Identification:**  
> Detailed step-by-step instructions and scripts on how to extract and identify the kinematic parameters (`x0, y0, z0, zt, l, h, alpha, beta, thigh_offset, knee_offset, model_knee_offset`) directly from CAD / URDF models for different overconstrained limb morphologies will be provided in an upcoming update.

---

## License

This project is released under the [MIT License](LICENSE).
