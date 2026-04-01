"""
Robot configuration definitions.
Defines DH parameters (Spong convention: [a, alpha, d, theta_offset])
and degrees of freedom for each robot.
"""
import numpy as np
import os
import tempfile


ROBOTS = {
    "kuka_kr6": {
        "name": "KUKA KR6",
        "dof": 6,
        "dh_params": np.array([
            [0.025,   -np.pi/2, 0.183, 0.0],
            [-0.315,   0.0,     0.0,   0.0],
            [-0.035,   np.pi/2, 0.0,   0.0],
            [0.0,     -np.pi/2, 0.365, 0.0],
            [0.0,      np.pi/2, 0.0,   0.0],
            [0.0,      0.0,     0.08,  0.0],
        ]),
        "description": "6-DOF industrial robot (spherical wrist; closed-form solution exists)",
    },
    "fanuc_crx10ia": {
        "name": "FANUC CRX-10iA",
        "dof": 6,
        "dh_params": np.array([
            [0.0,    -np.pi/2,  0.245,  0.0],
            [0.540,   np.pi,    0.0,   -np.pi/2],
            [0.0,    -np.pi/2,  0.0,    0.0],
            [0.0,     np.pi/2, -0.540,  0.0],
            [0.0,    -np.pi/2,  0.150,  0.0],
            [0.0,     0.0,     -0.160,  0.0],
        ]),
        "description": "6-DOF cobot with non-spherical wrist (0.15m offset; no closed-form solution)",
    },
    "panda": {
        "name": "Franka Panda",
        "dof": 7,
        "dh_params": np.array([
            [0.0,     -np.pi/2, 0.333,  0.0],
            [0.0,      np.pi/2, 0.0,    0.0],
            [0.0825,   np.pi/2, 0.316,  0.0],
            [-0.0825, -np.pi/2, 0.0,    0.0],
            [0.0,     -np.pi/2, 0.384,  0.0],
            [0.088,    np.pi/2, 0.0,    0.0],
            [0.0,      0.0,     0.107,  0.0],
        ]),
        "description": "7-DOF redundant robot (no closed-form solution; numerical methods required)",
    },
}


def get_robot(name: str) -> dict:
    """Get robot configuration by name."""
    if name not in ROBOTS:
        raise ValueError(f"Unknown robot: {name}. Available: {list(ROBOTS.keys())}")
    return ROBOTS[name]


def dh_transform(a, alpha, d, theta):
    """Compute homogeneous transform from a single DH parameter set (Spong convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,   0.0,  1.0],
    ])


def forward_kinematics(q, dh_params):
    """Forward kinematics: compute homogeneous transform from joint angles."""
    T = np.eye(4)
    for i in range(len(q)):
        a, alpha, d, theta_offset = dh_params[i]
        theta = theta_offset + q[i]
        T = T @ dh_transform(a, alpha, d, theta)
    return T


def jacobian(q, dh_params):
    """Compute 6×n geometric Jacobian.

    For revolute joint i (0-indexed), the Jacobian column is:
      J[:3, i] = z_i × (o_n - o_i)   (linear velocity)
      J[3:, i] = z_i                  (angular velocity)
    where z_i, o_i come from cumulative DH transform T_0^i.
    """
    n = len(q)
    T = np.eye(4)
    transforms = [T.copy()]
    for i in range(n):
        a, alpha, d, theta_offset = dh_params[i]
        theta = theta_offset + q[i]
        T = T @ dh_transform(a, alpha, d, theta)
        transforms.append(T.copy())
    o_n = transforms[n][:3, 3]
    J = np.zeros((6, n))
    for i in range(n):
        z = transforms[i][:3, 2]
        o = transforms[i][:3, 3]
        J[:3, i] = np.cross(z, o_n - o)
        J[3:, i] = z
    return J


def _count_aspect_crossings(q1, q2, dh_params, n_steps=30):
    """Count sign changes of det(J) along straight-line path from q1 to q2.

    Each sign change indicates the path crosses the singular surface det(J)=0.
    """
    crossings = 0
    prev_sign = np.sign(np.linalg.det(jacobian(q1, dh_params)))
    for k in range(1, n_steps + 1):
        t = k / n_steps
        q_t = (1 - t) * q1 + t * q2
        curr_sign = np.sign(np.linalg.det(jacobian(q_t, dh_params)))
        if curr_sign != prev_sign and prev_sign != 0 and curr_sign != 0:
            crossings += 1
        prev_sign = curr_sign
    return crossings


def compute_joint_metrics(q_solved, q_true, fk_errors, dh_params=None,
                          fk_threshold=1e-6):
    """Compute joint-space and branch classification metrics.

    Joint-space: checks ||q_solved - q_true|| (2π-wrapped) per joint.
    Branch (6-DOF only): classifies by counting det(J) sign changes along
    the straight-line path from q_true to q_solved in joint space.
      - same_aspect:    FK-ok AND 0 crossings (path stays in one aspect)
      - diff_aspect:    FK-ok AND ≥1 crossings (path crosses singular surface)
      - cuspidal_swap:  same_aspect but joint config differs (max|Δq| ≥ 0.1)
        → true cuspidal behavior: branch change without crossing det(J)=0

    Args:
        q_solved: (n, dof) solved joint configurations
        q_true: (n, dof) ground truth joint configurations
        fk_errors: (n,) FK errors per solve
        dh_params: DH parameter array (required for branch metrics on 6-DOF)
        fk_threshold: FK error threshold for FK-success gating

    Returns:
        dict with joint_match_rate, joint_err_median, and (for 6-DOF with dh_params)
        same_aspect_rate, diff_aspect_rate, cuspidal_swap_rate.
    """
    q_solved = np.asarray(q_solved, dtype=float)
    q_true = np.asarray(q_true, dtype=float)
    fk_errors = np.asarray(fk_errors, dtype=float)
    n = len(q_true)
    dof = q_true.shape[1]

    diff = q_solved - q_true
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    max_joint_err = np.max(np.abs(diff), axis=1)

    fk_ok = fk_errors < fk_threshold
    joint_match = fk_ok & (max_joint_err < 0.1)

    result = {
        "joint_match_rate": float(np.sum(joint_match)) / n,
    }
    if np.any(fk_ok):
        result["joint_err_median"] = float(np.median(max_joint_err[fk_ok]))
    else:
        result["joint_err_median"] = float("inf")

    # Branch classification via path crossing count (6-DOF only)
    if dh_params is not None and dof == 6:
        no_crossing = np.zeros(n, dtype=bool)
        for i in range(n):
            if not fk_ok[i]:
                continue
            nc = _count_aspect_crossings(q_true[i], q_solved[i], dh_params)
            no_crossing[i] = (nc == 0)

        same_aspect = fk_ok & no_crossing
        diff_aspect = fk_ok & ~no_crossing
        cuspidal_swap = same_aspect & (max_joint_err >= 0.1)

        result["same_aspect_rate"] = float(np.sum(same_aspect)) / n
        result["diff_aspect_rate"] = float(np.sum(diff_aspect)) / n
        result["cuspidal_swap_rate"] = float(np.sum(cuspidal_swap)) / n

    return result


def generate_test_poses(robot_name: str, n=100, seed=42, near_init=True, mode=None):
    """
    Generate test joint-angle / target-pose pairs.
    Args:
        robot_name: Robot name
        n: Number of problems
        seed: Random seed
        near_init: Legacy param (True=near, False=random). Ignored when mode is set.
        mode: Init-guess mode string (overrides near_init when provided):
              "near"      - q_true ± 0.1  (local convergence test)
              "far"       - q_true ± 1.0  (medium-distance convergence)
              "j1_offset" - q_true with only J1 offset by ±π
              "zeros"     - all-zero initial guess (home position)
              "random"    - U[-π, π]  (global search)
    Returns:
        q_true: (n, DOF)
        q_init: (n, DOF)
        targets: (n, 4, 4)
    """
    if mode is None:
        mode = "near" if near_init else "random"

    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]

    rng = np.random.default_rng(seed)
    q_true = rng.uniform(-1.0, 1.0, size=(n, dof))

    if mode == "near":
        q_init = q_true + rng.uniform(-0.1, 0.1, size=(n, dof))
    elif mode == "far":
        q_init = q_true + rng.uniform(-1.0, 1.0, size=(n, dof))
    elif mode == "j1_offset":
        q_init = q_true.copy()
        q_init[:, 0] += rng.uniform(-np.pi, np.pi, size=n)
    elif mode == "zeros":
        q_init = np.zeros((n, dof))
    elif mode == "random":
        q_init = rng.uniform(-np.pi, np.pi, size=(n, dof))
    else:
        raise ValueError(f"Unknown mode: {mode}. "
                         f"Use: near, far, j1_offset, zeros, random")

    targets = np.array([forward_kinematics(q, dh) for q in q_true])
    return q_true, q_init, targets


def _rotation_to_rpy(R):
    """Decompose rotation matrix to (roll, pitch, yaw) for URDF.

    URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    sp = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)
    if abs(sp) < 1.0 - 1e-10:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = 0.0
        roll = np.arctan2(R[0, 1], R[0, 2]) if sp > 0 else np.arctan2(-R[0, 1], -R[0, 2])
    return roll, pitch, yaw


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def generate_urdf(robot_name: str) -> str:
    """Generate a URDF string from DH parameters.

    DH convention: T_i = Rz(q_i + θ_off_i) * Tz(d_i) * Tx(a_i) * Rx(α_i)

    URDF mapping (supports non-zero theta_offset):
      joint_1:        origin = Rz(θ_off_0), axis = z
      joint_i (i>1):  origin = Tz(d_{i-1}) * Tx(a_{i-1}) * Rx(α_{i-1}) * Rz(θ_off_i)
      ee_joint:       origin = Tz(d_n) * Tx(a_n) * Rx(α_n)
    """
    robot = get_robot(robot_name)
    dh = robot["dh_params"]
    dof = robot["dof"]

    def _origin_str(R, t):
        r, p, y = _rotation_to_rpy(R)
        xyz = " ".join(f"{v:.15g}" for v in t)
        rpy = " ".join(f"{v:.15g}" for v in [r, p, y])
        return xyz, rpy

    lines = ['<?xml version="1.0" ?>', f'<robot name="{robot_name}">',
             '  <link name="base_link"/>']
    for i in range(dof):
        if i == 0:
            R = _rotz(dh[0, 3])
            t = np.zeros(3)
        else:
            a_prev, alpha_prev, d_prev, _ = dh[i - 1]
            theta_off_i = dh[i, 3]
            R = _rotx(alpha_prev) @ _rotz(theta_off_i)
            t = np.array([a_prev, 0.0, d_prev])
        xyz, rpy = _origin_str(R, t)
        parent = "base_link" if i == 0 else f"link_{i}"
        lines += [
            f'  <link name="link_{i+1}"/>',
            f'  <joint name="joint_{i+1}" type="revolute">',
            f'    <parent link="{parent}"/>',
            f'    <child link="link_{i+1}"/>',
            f'    <origin xyz="{xyz}" rpy="{rpy}"/>',
            f'    <axis xyz="0 0 1"/>',
            f'    <limit lower="-6.28" upper="6.28" effort="100" velocity="3.14"/>',
            f'  </joint>',
        ]
    la, lalpha, ld, _ = dh[-1]
    R_ee = _rotx(lalpha)
    t_ee = np.array([la, 0.0, ld])
    xyz, rpy = _origin_str(R_ee, t_ee)
    lines += [
        '  <link name="ee_link"/>',
        '  <joint name="ee_joint" type="fixed">',
        f'    <parent link="link_{dof}"/>',
        '    <child link="ee_link"/>',
        f'    <origin xyz="{xyz}" rpy="{rpy}"/>',
        '  </joint>',
        '</robot>',
    ]
    return '\n'.join(lines)


def get_urdf_path(robot_name: str) -> str:
    """Get path to a cached URDF file, creating it if needed."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".urdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{robot_name}.urdf")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(generate_urdf(robot_name))
    return path


# ---------------------------------------------------------------------------
# Linear-path benchmark helpers
# ---------------------------------------------------------------------------

def generate_linear_paths(robot_name: str, n_paths: int = 50,
                          n_waypoints: int = 21, line_length: float = 0.15,
                          seed: int = 42):
    """Generate straight-line Cartesian paths for linearity benchmarks.

    Each path starts from a random reachable configuration.  The end-effector
    position is translated linearly along a random direction while keeping the
    orientation constant.

    Args:
        robot_name: Robot identifier.
        n_paths: Number of independent paths.
        n_waypoints: Waypoints per path (including start and end).
        line_length: Path length in metres (0.10–0.20 recommended).
        seed: Random seed.

    Returns:
        list of dicts, one per path, each containing:
            targets   – (n_waypoints, 4, 4) target poses
            q_seed    – (dof,) starting joint config (near-start seed)
            ideal_pos – (n_waypoints, 3) ideal Cartesian positions
    """
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    rng = np.random.default_rng(seed)

    paths = []
    for _ in range(n_paths):
        q_start = rng.uniform(-1.0, 1.0, size=dof)
        T_start = forward_kinematics(q_start, dh)
        p_start = T_start[:3, 3].copy()
        R_start = T_start[:3, :3].copy()

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        targets = np.empty((n_waypoints, 4, 4))
        ideal_pos = np.empty((n_waypoints, 3))
        for k in range(n_waypoints):
            t = k / (n_waypoints - 1)
            p = p_start + direction * (line_length * t)
            T = np.eye(4)
            T[:3, :3] = R_start
            T[:3, 3] = p
            targets[k] = T
            ideal_pos[k] = p

        # Seed for first waypoint: near the true start config
        q_seed = q_start + rng.uniform(-0.1, 0.1, size=dof)

        paths.append({
            "targets": targets,
            "q_seed": q_seed,
            "ideal_pos": ideal_pos,
        })
    return paths


def compute_linearity(fk_positions: np.ndarray,
                      ideal_positions: np.ndarray) -> dict:
    """Measure deviation of FK results from the ideal straight line.

    For each path the ideal line runs from ideal_positions[0] to
    ideal_positions[-1].  Deviation is the perpendicular distance from each
    intermediate FK position to this line.

    Args:
        fk_positions: (n_waypoints, 3) realised end-effector positions.
        ideal_positions: (n_waypoints, 3) target positions on the line.

    Returns:
        dict with max_dev and mean_dev (metres).
    """
    p0 = ideal_positions[0]
    p1 = ideal_positions[-1]
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-12:
        return {"max_dev": 0.0, "mean_dev": 0.0}
    d_hat = d / length

    devs = []
    for p in fk_positions:
        v = p - p0
        proj = np.dot(v, d_hat) * d_hat
        dev = np.linalg.norm(v - proj)
        devs.append(dev)
    devs = np.array(devs)
    return {
        "max_dev": float(np.max(devs)),
        "mean_dev": float(np.mean(devs)),
    }
