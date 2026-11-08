import os
import time
import math
import torch
import numpy as np
import yourdfpy
import viser
from viser.extras import ViserUrdf

from ov_kin.as_over_dog_open_kinematics import (
    AsOvDogOpenBennettSolver,
    AsOvDogOpenPlanarSolver,
    AsOvDogOpenSphericalSolver,
)

# ---------------------------------------------------------------------------
# 1. User-Configurable Default Standing Actuator Positions (Shape: 4 x 3)
#    Order of limbs: [FL, FR, RL, RR], Order of joints: [hip, thigh, calf]
# ---------------------------------------------------------------------------
DEFAULT_STANDING_ACT_POS = {
    "Bennett": [
        [-0.67,  1.91, -0.74],  # FL
        [0.67, -1.91,  0.74],  # FR
        [0.67, -1.91,  0.74],  # RL
        [-0.67,  1.91, -0.74],  # RR
    ],
    "Planar": [
        [-1.38,  2.39, -1.52],  # FL
        [1.38, -2.39,  1.52],  # FR
        [1.38, -2.39,  1.52],  # RL
        [-1.38,  2.39, -1.52],  # RR
    ],
    "Spherical": [
        [-0.94,  2.43, -1.57],  # FL
        [0.94, -2.43,  1.57],  # FR
        [0.94, -2.43,  1.57],  # RL
        [-0.94,  2.43, -1.57],  # RR
    ],
}

# Physical actuator joint limits from the URDF files: (min_rad, max_rad)
LEGS = ["FL", "FR", "RL", "RR"]
ACTUATOR_LIMITS = {
    "FL": {"hip": (-math.pi, 0.5236), "thigh": (0.0, 4.71), "calf": (-2.617994, 0.0)},
    "FR": {"hip": (-0.5236, math.pi), "thigh": (-4.71, 0.0), "calf": (0.0, 2.617994)},
    "RL": {"hip": (-0.5236, math.pi), "thigh": (-4.71, 0.0), "calf": (0.0, 2.617994)},
    "RR": {"hip": (-math.pi, 0.5236), "thigh": (0.0, 4.71), "calf": (-2.617994, 0.0)},
}

# Pre-build tensors of lower/upper actuator bounds of shape (1, 4, 3) for fast clamping
ACT_LOWER_BOUNDS = torch.tensor(
    [[
        [ACTUATOR_LIMITS[leg]["hip"][0], ACTUATOR_LIMITS[leg]["thigh"][0], ACTUATOR_LIMITS[leg]["calf"][0]]
        for leg in LEGS
    ]],
    dtype=torch.float64,
)
ACT_UPPER_BOUNDS = torch.tensor(
    [[
        [ACTUATOR_LIMITS[leg]["hip"][1], ACTUATOR_LIMITS[leg]["thigh"][1], ACTUATOR_LIMITS[leg]["calf"][1]]
        for leg in LEGS
    ]],
    dtype=torch.float64,
)

# ---------------------------------------------------------------------------
# 2. Pre-load Solvers and URDF Models on CPU
# ---------------------------------------------------------------------------
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")

SOLVERS = {
    "Bennett": AsOvDogOpenBennettSolver(device="cpu", dtype=torch.float64, output_dtype=torch.float64),
    "Planar": AsOvDogOpenPlanarSolver(device="cpu", dtype=torch.float64, output_dtype=torch.float64),
    "Spherical": AsOvDogOpenSphericalSolver(device="cpu", dtype=torch.float64, output_dtype=torch.float64),
}

URDF_PATHS = {
    "Bennett": os.path.join(ASSET_DIR, "AsOverDogOpenBennett.urdf"),
    "Planar": os.path.join(ASSET_DIR, "AsOverDogOpenPlanar.urdf"),
    "Spherical": os.path.join(ASSET_DIR, "AsOverDogOpenSpherical.urdf"),
}

URDF_MODELS = {
    name: yourdfpy.URDF.load(path, mesh_dir=ASSET_DIR)
    for name, path in URDF_PATHS.items()
}


def get_default_act_tensor(morph_name: str) -> torch.Tensor:
    """Returns a fresh (1, 4, 3) tensor of the default standing actuator positions."""
    return torch.tensor([DEFAULT_STANDING_ACT_POS[morph_name]], dtype=torch.float64)


def compute_urdf_joint_dict(solver, act_pos: torch.Tensor) -> dict[str, float]:
    """
    Converts a (1, 4, 3) actuator position tensor into the 20 revolute joint angles
    expected by AsOverDogOpen{Bennett,Planar,Spherical}.urdf using knee_pos_from_actuator.
    """
    act_np = act_pos[0].detach().cpu().numpy()  # Shape: (4, 3)
    # knee_pos_from_actuator returns shape (4,) for batch_size=1 when use_urdf_convention=True
    knee_np = solver.knee_pos_from_actuator(act_pos, use_urdf_convention=True).detach().cpu().numpy().reshape(-1)

    joint_cfg = {}
    for i, leg in enumerate(LEGS):
        knee_angle = float(knee_np[i])
        joint_cfg[f"motor_hip_{leg}"] = float(act_np[i, 0])
        joint_cfg[f"motor_thigh_{leg}"] = float(act_np[i, 1])
        joint_cfg[f"motor_calf_{leg}"] = float(act_np[i, 2])
        joint_cfg[f"knee_real_{leg}"] = knee_angle
        joint_cfg[f"knee_oppo_{leg}"] = knee_angle

    return joint_cfg


# ---------------------------------------------------------------------------
# 3. Start Viser Server & Per-Visitor Session Handler
# ---------------------------------------------------------------------------
server = viser.ViserServer(host="0.0.0.0", port=7860, label="AsOverDog Kinematics")


@server.on_client_connect
def handle_new_client(client: viser.ClientHandle) -> None:
    """Each browser visitor gets their own isolated 3D scene & GUI state."""
    client.camera.position = (0.75, -0.75, 0.45)
    client.camera.look_at = (0.0, 0.0, -0.05)

    client.scene.add_grid("/ground", width=2.0, height=2.0, position=(0.0, 0.0, -0.32))

    state = {
        "morph": "Bennett",
        "updating": False,
        "act_pos": get_default_act_tensor("Bennett"),
        "viser_urdf": None,
    }

    def load_urdf_for_client(morph_name: str):
        if state["viser_urdf"] is not None:
            state["viser_urdf"].remove()
        state["viser_urdf"] = ViserUrdf(
            client,
            urdf_or_path=URDF_MODELS[morph_name],
            root_node_name="/robot",
        )

    load_urdf_for_client(state["morph"])

    # --- Top GUI Controls ---
    with client.gui.add_folder("Robot Morphology & Status"):
        morph_dropdown = client.gui.add_dropdown(
            "Morphology",
            options=["Bennett", "Planar", "Spherical"],
            initial_value="Bennett",
        )
        status_md = client.gui.add_markdown("**Status:** All 4 limbs in workspace ✅")
        reset_btn = client.gui.add_button("Reset to Standing Pose")

    act_sliders = {}
    foot_sliders = {}
    foot_gizmos = {}

    solver = SOLVERS[state["morph"]]
    init_fk = solver.fk(state["act_pos"])[0].detach().cpu().numpy()

    # --- Per-Leg Folders & 3D Transform Gizmos ---
    for i, leg in enumerate(LEGS):
        hip_lim = ACTUATOR_LIMITS[leg]["hip"]
        thigh_lim = ACTUATOR_LIMITS[leg]["thigh"]
        calf_lim = ACTUATOR_LIMITS[leg]["calf"]

        with client.gui.add_folder(f"Limb {leg}", expand_by_default=(i == 0)):
            q1_s = client.gui.add_slider(
                f"{leg} Hip (q1) [rad]", min=hip_lim[0], max=hip_lim[1],
                step=0.005, initial_value=float(state["act_pos"][0, i, 0])
            )
            q2_s = client.gui.add_slider(
                f"{leg} Thigh (q2) [rad]", min=thigh_lim[0], max=thigh_lim[1],
                step=0.005, initial_value=float(state["act_pos"][0, i, 1])
            )
            q3_s = client.gui.add_slider(
                f"{leg} Calf (q3) [rad]", min=calf_lim[0], max=calf_lim[1],
                step=0.005, initial_value=float(state["act_pos"][0, i, 2])
            )
            act_sliders[leg] = (q1_s, q2_s, q3_s)

            fx_s = client.gui.add_slider(
                f"{leg} Foot X [m]", min=-0.65, max=0.65,
                step=0.001, initial_value=float(init_fk[i, 0])
            )
            fy_s = client.gui.add_slider(
                f"{leg} Foot Y [m]", min=-0.40, max=0.40,
                step=0.001, initial_value=float(init_fk[i, 1])
            )
            fz_s = client.gui.add_slider(
                f"{leg} Foot Z [m]", min=-0.45, max=0.25,
                step=0.001, initial_value=float(init_fk[i, 2])
            )
            foot_sliders[leg] = (fx_s, fy_s, fz_s)

        # Add interactive 3D XYZ draggable gizmo at the FootTip position
        gizmo = client.scene.add_transform_controls(
            f"/foot_target_{leg}",
            position=tuple(init_fk[i]),
            scale=0.075,
            disable_rotations=True,
        )
        foot_gizmos[leg] = gizmo

        # --- Callbacks for this leg ---
        def make_fk_callback(leg_idx=i, leg_name=leg):
            def _on_act_change(_event):
                if state["updating"]:
                    return
                state["updating"] = True
                try:
                    q1_s, q2_s, q3_s = act_sliders[leg_name]
                    state["act_pos"][0, leg_idx, 0] = q1_s.value
                    state["act_pos"][0, leg_idx, 1] = q2_s.value
                    state["act_pos"][0, leg_idx, 2] = q3_s.value
                    sync_from_actuators(status_text=f"**Mode:** Forward Kinematics ({leg_name})")
                finally:
                    state["updating"] = False
            return _on_act_change

        for s in act_sliders[leg]:
            s.on_update(make_fk_callback())

        def make_ik_slider_callback(leg_idx=i, leg_name=leg):
            def _on_foot_slider(_event):
                if state["updating"]:
                    return
                state["updating"] = True
                try:
                    fx_s, fy_s, fz_s = foot_sliders[leg_name]
                    target_pos = np.array([fx_s.value, fy_s.value, fz_s.value], dtype=np.float64)
                    run_ik_for_leg(leg_idx, target_pos, update_gizmo=True)
                finally:
                    state["updating"] = False
            return _on_foot_slider

        for s in foot_sliders[leg]:
            s.on_update(make_ik_slider_callback())

        def make_gizmo_callback(leg_idx=i, leg_name=leg):
            def _on_gizmo_drag(_event):
                if state["updating"]:
                    return
                state["updating"] = True
                try:
                    g_pos = np.array(foot_gizmos[leg_name].position, dtype=np.float64)
                    run_ik_for_leg(leg_idx, g_pos, update_gizmo=False)
                finally:
                    state["updating"] = False
            return _on_gizmo_drag

        gizmo.on_update(make_gizmo_callback())

    # --- Synchronization Helpers ---
    def sync_from_actuators(status_text: str | None = None, update_gizmo: bool = True):
        cur_solver = SOLVERS[state["morph"]]
        # 1. Update URDF 3D meshes using knee_pos_from_actuator
        joint_cfg = compute_urdf_joint_dict(cur_solver, state["act_pos"])
        state["viser_urdf"].update_cfg(joint_cfg)

        # 2. Compute FK and update foot sliders & 3D gizmos
        fk_all = cur_solver.fk(state["act_pos"])[0].detach().cpu().numpy()
        for idx, leg in enumerate(LEGS):
            q1_s, q2_s, q3_s = act_sliders[leg]
            q1_s.value = float(state["act_pos"][0, idx, 0])
            q2_s.value = float(state["act_pos"][0, idx, 1])
            q3_s.value = float(state["act_pos"][0, idx, 2])

            fx_s, fy_s, fz_s = foot_sliders[leg]
            fx_s.value = float(fk_all[idx, 0])
            fy_s.value = float(fk_all[idx, 1])
            fz_s.value = float(fk_all[idx, 2])

            if update_gizmo:
                foot_gizmos[leg].position = tuple(fk_all[idx])

        if status_text is not None:
            status_md.content = status_text

    def run_ik_for_leg(leg_idx: int, target_xyz: np.ndarray, update_gizmo: bool = True):
        cur_solver = SOLVERS[state["morph"]]
        all_feet = cur_solver.fk(state["act_pos"]).clone()
        all_feet[0, leg_idx] = torch.tensor(target_xyz, dtype=torch.float64)

        act_sol, ik_errors, oow = cur_solver.ik(all_feet, state["act_pos"])
        act_sol = act_sol.to(dtype=torch.float64)

        # Check if the solved actuator angles violate the physical URDF actuator limits (e.g. calf limit)
        raw_leg_act = act_sol[0, leg_idx]
        min_b = ACT_LOWER_BOUNDS[0, leg_idx]
        max_b = ACT_UPPER_BOUNDS[0, leg_idx]
        limit_violated = bool(torch.any((raw_leg_act < min_b - 1e-4) | (raw_leg_act > max_b + 1e-4)).item())

        # Clamp actuator positions to physical URDF limits so the robot never exceeds joint limits
        state["act_pos"] = torch.clamp(act_sol, min=ACT_LOWER_BOUNDS, max=ACT_UPPER_BOUNDS)

        # Recompute actual FK error after clamping to actuator limits
        actual_fk = cur_solver.fk(state["act_pos"])[0, leg_idx].detach().cpu().numpy()
        actual_err_mm = float(np.linalg.norm(actual_fk - target_xyz)) * 1000.0
        is_oow = bool(oow[0, leg_idx].item())
        leg_name = LEGS[leg_idx]

        if is_oow:
            msg = f"**Status:** ⚠️ `{leg_name}` Out of Workspace! (Residual Error: `{actual_err_mm:.2f} mm`)"
        elif limit_violated:
            msg = f"**Status:** ⚠️ `{leg_name}` Hit Actuator Joint Limit! (Clamped Error: `{actual_err_mm:.2f} mm`)"
        else:
            msg = f"**Status:** ✅ `{leg_name}` IK Solved (FK Error: `{actual_err_mm:.4f} mm`)"

        sync_from_actuators(status_text=msg, update_gizmo=update_gizmo)

    @morph_dropdown.on_update
    def _(_event):
        if state["updating"]:
            return
        state["updating"] = True
        try:
            state["morph"] = morph_dropdown.value
            state["act_pos"] = get_default_act_tensor(state["morph"])
            load_urdf_for_client(state["morph"])
            sync_from_actuators(status_text=f"**Switched to:** `{state['morph']}`")
        finally:
            state["updating"] = False

    @reset_btn.on_click
    def _(_event):
        state["updating"] = True
        try:
            state["act_pos"] = get_default_act_tensor(state["morph"])
            sync_from_actuators(status_text="**Status:** Reset to standing pose ✅")
        finally:
            state["updating"] = False

    # Initial render for this client
    state["updating"] = True
    sync_from_actuators()
    state["updating"] = False


if __name__ == "__main__":
    print("Viser server running at http://localhost:7860")
    while True:
        time.sleep(1.0)