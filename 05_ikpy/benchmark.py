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
from common.robots import get_robot, compute_joint_metrics, compute_linearity


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


def generate_test_poses_ikpy(chain, dof, n=100, seed=42, mode="near"):
    """Generate test poses using IKPy's own FK (ensures FK consistency)."""
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
        raise ValueError(f"Unknown mode: {mode}")
    targets = []
    for i in range(n):
        full_q = np.concatenate([[0.0], q_true[i]])
        T = chain.forward_kinematics(full_q)
        targets.append(T)
    return q_true, q_init, np.array(targets)


def generate_linear_paths_ikpy(chain, dof, n_paths=50, n_waypoints=21,
                                line_length=0.15, seed=42):
    """Generate linear paths using IKPy's own FK for consistency."""
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_paths):
        q_start = rng.uniform(-1.0, 1.0, size=dof)
        full_q = np.concatenate([[0.0], q_start])
        T_start = chain.forward_kinematics(full_q)
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

        q_seed = q_start + rng.uniform(-0.1, 0.1, size=dof)
        paths.append({
            "targets": targets,
            "q_seed": q_seed,
            "ideal_pos": ideal_pos,
        })
    return paths


def run_benchmark(n=100, robot_name="kuka_kr6", mode="near", timeout_us=0):
    if mode == "linear":
        return run_linear_benchmark(n, robot_name, timeout_us=timeout_us)
    robot = get_robot(robot_name)
    dof = robot["dof"]
    chain = build_chain(robot_name)
    q_true, q_init, targets = generate_test_poses_ikpy(chain, dof, n, mode=mode)

    errors = []
    q_solved_list = []
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
        q_solved_list.append(q_result[1:])  # skip fixed base link

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
    jm = compute_joint_metrics(np.array(q_solved_list), q_true, errors,
                              dh_params=robot["dh_params"], fk_threshold=1e-3)
    print(f"joint_match_rate: {jm['joint_match_rate']:.4f}")
    print(f"joint_err_median: {jm['joint_err_median']:.2e}")
    if "same_aspect_rate" in jm:
        print(f"same_aspect_rate: {jm['same_aspect_rate']:.4f}")
        print(f"diff_aspect_rate: {jm['diff_aspect_rate']:.4f}")
        print(f"cuspidal_swap_rate: {jm['cuspidal_swap_rate']:.4f}")


def run_linear_benchmark(n_paths, robot_name, timeout_us=0):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    chain = build_chain(robot_name)
    paths = generate_linear_paths_ikpy(chain, dof, n_paths=n_paths)
    timeout_s = timeout_us * 1e-6 if timeout_us > 0 else 0

    all_errors = []
    all_devs = []
    timeout_count = 0
    total_start = time.perf_counter()
    for path in paths:
        targets = path["targets"]
        q_cur = path["q_seed"].copy()
        fk_pos = []
        for T in targets:
            t0 = time.perf_counter()
            initial = np.concatenate([[0.0], q_cur])
            q_result = chain.inverse_kinematics_frame(
                T, initial_position=initial, orientation_mode="all",
            )
            dt = time.perf_counter() - t0
            if timeout_s > 0 and dt > timeout_s:
                timeout_count += 1
                all_errors.append(float("inf"))
                fk_pos.append(T[:3, 3])
            else:
                q_sol = q_result[1:]
                fk_result = chain.forward_kinematics(q_result)
                pos_err = np.linalg.norm(fk_result[:3, 3] - T[:3, 3])
                rot_err = np.linalg.norm(fk_result[:3, :3] - T[:3, :3])
                all_errors.append(np.sqrt(pos_err**2 + rot_err**2))
                fk_pos.append(fk_result[:3, 3])
                q_cur = q_sol
        lin = compute_linearity(np.array(fk_pos), path["ideal_pos"])
        all_devs.append(lin)
    total_elapsed = time.perf_counter() - total_start

    errors = np.array(all_errors)
    n_total = len(errors)
    success = np.sum(errors < 1e-3)
    total_us = total_elapsed * 1e6
    max_devs = [d["max_dev"] for d in all_devs]
    mean_devs = [d["mean_dev"] for d in all_devs]
    print(f"method: ikpy")
    print(f"num_solves: {n_total}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n_total:.1f}")
    print(f"success_rate: {success/n_total:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"median_error: {np.median(errors):.2e}")
    print(f"timeout_count: {timeout_count}")
    print(f"timeout_rate: {timeout_count/n_total:.4f}")
    print(f"linearity_max_dev: {np.max(max_devs):.2e}")
    print(f"linearity_mean_dev: {np.mean(mean_devs):.2e}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    timeout_us = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    run_benchmark(n, robot_name=robot, mode=mode, timeout_us=timeout_us)
