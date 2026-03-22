"""
IK benchmark using the IKPy library.
IKPy builds URDF-link-based chains and solves IK via scipy optimization.

Note: IKPy's URDFLink coordinate convention differs from standard DH,
so target poses are generated and results verified using IKPy's own FK.
"""
import time
import sys
import os
import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.robots import get_robot


def build_chain(robot_name: str):
    """Build an IKPy chain from DH parameters for the given robot."""
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    links = [OriginLink()]
    for i in range(dof):
        a, alpha, d, theta_offset = dh[i]
        links.append(URDFLink(
            name=f"joint_{i+1}",
            origin_translation=[a, 0, d],
            origin_orientation=[alpha, 0, theta_offset],
            rotation=[0, 0, 1],
        ))
    return Chain(name=robot_name, links=links, active_links_mask=[False] + [True]*dof)


def generate_test_poses_ikpy(chain, dof, n=100, seed=42, near_init=True):
    """Generate test poses using IKPy's own FK (ensures FK consistency)."""
    rng = np.random.default_rng(seed)
    q_true = rng.uniform(-1.0, 1.0, size=(n, dof))
    if near_init:
        q_init = q_true + rng.uniform(-0.1, 0.1, size=(n, dof))
    else:
        q_init = rng.uniform(-np.pi, np.pi, size=(n, dof))
    targets = []
    for i in range(n):
        full_q = np.concatenate([[0.0], q_true[i]])
        T = chain.forward_kinematics(full_q)
        targets.append(T)
    return q_true, q_init, np.array(targets)


def run_benchmark(n=100, robot_name="kuka_kr6", near_init=True):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    chain = build_chain(robot_name)
    q_true, q_init, targets = generate_test_poses_ikpy(chain, dof, n, near_init=near_init)

    errors = []
    total_start = time.perf_counter()

    for i in range(n):
        target_pos = targets[i][:3, 3]
        target_orientation = targets[i][:3, :3]
        initial = np.concatenate([[0.0], q_init[i]])
        q_result = chain.inverse_kinematics_frame(
            targets[i],
            initial_position=initial,
            orientation_mode="all",
        )
        fk_result = chain.forward_kinematics(q_result)
        pos_err = np.linalg.norm(fk_result[:3, 3] - target_pos)
        rot_err = np.linalg.norm(fk_result[:3, :3] - target_orientation)
        errors.append(np.sqrt(pos_err**2 + rot_err**2))

    total_elapsed = time.perf_counter() - total_start

    errors = np.array(errors)
    success = np.sum(errors < 1e-3)
    total_us = total_elapsed * 1e6
    print(f"method: ikpy")
    print(f"num_solves: {n}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n:.1f}")
    print(f"success_rate: {success/n:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"mean_error: {np.mean(errors):.2e}")
    print(f"median_error: {np.median(errors):.2e}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    run_benchmark(n, robot_name=robot, near_init=(mode != "random"))
