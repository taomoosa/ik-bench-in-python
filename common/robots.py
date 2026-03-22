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


def generate_test_poses(robot_name: str, n=100, seed=42, near_init=True):
    """
    Generate test joint-angle / target-pose pairs.
    Args:
        robot_name: Robot name ("kuka_kr6" or "panda")
        n: Number of problems
        seed: Random seed
        near_init: True = initial guess near solution (±0.1), False = fully random
    Returns:
        q_true: (n, DOF)
        q_init: (n, DOF)
        targets: (n, 4, 4)
    """
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]

    rng = np.random.default_rng(seed)
    q_true = rng.uniform(-1.0, 1.0, size=(n, dof))
    if near_init:
        q_init = q_true + rng.uniform(-0.1, 0.1, size=(n, dof))
    else:
        q_init = rng.uniform(-np.pi, np.pi, size=(n, dof))
    targets = np.array([forward_kinematics(q, dh) for q in q_true])
    return q_true, q_init, targets


def generate_urdf(robot_name: str) -> str:
    """Generate a URDF string from DH parameters.

    DH convention: T_i = Rz(q_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

    URDF mapping (all theta_offset assumed 0):
      joint_1: origin = identity, axis = z
      joint_i (i>1): origin xyz="a_{i-1} 0 d_{i-1}" rpy="alpha_{i-1} 0 0"
      ee_joint (fixed): origin xyz="a_n 0 d_n" rpy="alpha_n 0 0"
    """
    robot = get_robot(robot_name)
    dh = robot["dh_params"]
    dof = robot["dof"]
    lines = ['<?xml version="1.0" ?>', f'<robot name="{robot_name}">',
             '  <link name="base_link"/>']
    for i in range(dof):
        if i == 0:
            xyz, rpy = "0 0 0", "0 0 0"
        else:
            pa, palpha, pd, _ = dh[i - 1]
            xyz, rpy = f"{pa} 0 {pd}", f"{palpha} 0 0"
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
    lines += [
        '  <link name="ee_link"/>',
        '  <joint name="ee_joint" type="fixed">',
        f'    <parent link="link_{dof}"/>',
        '    <child link="ee_link"/>',
        f'    <origin xyz="{la} 0 {ld}" rpy="{lalpha} 0 0"/>',
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
