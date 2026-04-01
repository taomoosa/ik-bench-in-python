"""
IK benchmark using OptIK (optik-py).
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.robots import (get_robot, generate_test_poses, get_urdf_path,
                           compute_joint_metrics, generate_linear_paths,
                           compute_linearity, forward_kinematics)


def _make_solver(urdf_path):
    import optik
    robot = optik.Robot.from_urdf_file(urdf_path, "base_link", "ee_link")
    robot.set_parallelism(1)
    config = optik.SolverConfig(solution_mode='quality', max_time=0.05,
                                tol_f=1e-10)
    return robot, config


def run_benchmark(n=100, robot_name="kuka_kr6", mode="near", timeout_us=0):
    if mode == "linear":
        return run_linear_benchmark(n, robot_name, timeout_us=timeout_us)
    robot = get_robot(robot_name)
    dof = robot["dof"]
    urdf_path = get_urdf_path(robot_name)
    optik_robot, config = _make_solver(urdf_path)

    q_true, q_init, targets = generate_test_poses(robot_name, n, mode=mode)

    errors = []
    q_solved_list = []
    total_start = time.perf_counter()

    for i in range(n):
        T = targets[i]
        T_list = T.tolist()
        result = optik_robot.ik(config, T_list, q_init[i].tolist())
        if result is not None:
            q_sol, residual = result
            T_sol = forward_kinematics(np.array(q_sol), robot["dh_params"])
            err = np.linalg.norm(T_sol[:3, :] - T[:3, :])
            errors.append(err)
            q_solved_list.append(np.array(q_sol))
        else:
            errors.append(float("inf"))
            q_solved_list.append(np.zeros(dof))

    total_elapsed = time.perf_counter() - total_start

    errors = np.array(errors)
    success = np.sum(errors < 1e-3)
    total_us = total_elapsed * 1e6
    print(f"method: optik")
    print(f"num_solves: {n}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n:.1f}")
    print(f"success_rate: {success/n:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"mean_error: {np.mean(errors[errors < 1e-3]):.2e}" if success > 0 else "mean_error: inf")
    print(f"median_error: {np.median(errors[errors < 1e-3]):.2e}" if success > 0 else "median_error: inf")
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
    dh = robot["dh_params"]
    urdf_path = get_urdf_path(robot_name)
    optik_robot, config = _make_solver(urdf_path)
    # Use speed mode with native timeout for time-constrained benchmarks
    if timeout_us > 0:
        import optik
        config = optik.SolverConfig(solution_mode='speed',
                                    max_time=timeout_us * 1e-6, tol_f=1e-6)
    paths = generate_linear_paths(robot_name, n_paths=n_paths)
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
            result = optik_robot.ik(config, T.tolist(), q_cur.tolist())
            dt = time.perf_counter() - t0
            if timeout_s > 0 and dt > timeout_s:
                timeout_count += 1
                all_errors.append(float("inf"))
                fk_pos.append(T[:3, 3])
            elif result is not None:
                q_sol, residual = result
                q_arr = np.array(q_sol)
                T_fk = forward_kinematics(q_arr, dh)
                err = np.linalg.norm(T_fk[:3, :] - T[:3, :])
                all_errors.append(err)
                fk_pos.append(T_fk[:3, 3])
                q_cur = q_arr
            else:
                all_errors.append(float("inf"))
                fk_pos.append(T[:3, 3])
        lin = compute_linearity(np.array(fk_pos), path["ideal_pos"])
        all_devs.append(lin)
    total_elapsed = time.perf_counter() - total_start

    errors = np.array(all_errors)
    n_total = len(errors)
    success = np.sum(errors < 1e-3)
    total_us = total_elapsed * 1e6
    max_devs = [d["max_dev"] for d in all_devs]
    mean_devs = [d["mean_dev"] for d in all_devs]
    print(f"method: optik")
    print(f"num_solves: {n_total}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n_total:.1f}")
    print(f"success_rate: {success/n_total:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"median_error: {np.median(errors[errors < 1e-3]):.2e}" if success > 0 else "median_error: inf")
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
