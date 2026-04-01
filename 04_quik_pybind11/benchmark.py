"""
Benchmark calling the quik IK solver via pybind11.
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
from common.robots import (get_robot, generate_test_poses, compute_joint_metrics,
                           generate_linear_paths, compute_linearity, forward_kinematics)

import quik_binding


def _make_solver(dh, dof):
    Cls = quik_binding.QuikSolver6 if dof == 6 else quik_binding.QuikSolver7
    return Cls(dh, max_iterations=200, max_consecutive_grad_fails=10,
               max_gradient_fails=80, lambda_squared=1e-10,
               max_linear_step_size=0.34)


def run_benchmark(n=100, robot_name="kuka_kr6", mode="near", timeout_us=0):
    if mode == "linear":
        return run_linear_benchmark(n, robot_name, timeout_us=timeout_us)
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    q_true, q_init, targets = generate_test_poses(robot_name, n, mode=mode)

    solver = _make_solver(dh, dof)

    results_err = []
    q_stars = []
    total_start = time.perf_counter()
    for i in range(n):
        q_star_i, err_i, iter_i = solver.ik(targets[i], q_init[i])
        results_err.append(err_i)
        q_stars.append(q_star_i)
    total_elapsed = time.perf_counter() - total_start

    errors = np.array(results_err)
    success = np.sum(errors < 1e-6)
    total_us = total_elapsed * 1e6
    print(f"method: quik_pybind11")
    print(f"num_solves: {n}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n:.1f}")
    print(f"success_rate: {success/n:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"mean_error: {np.mean(errors):.2e}")
    print(f"median_error: {np.median(errors):.2e}")
    jm = compute_joint_metrics(np.array(q_stars), q_true, errors,
                              dh_params=robot["dh_params"], fk_threshold=1e-6)
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
    paths = generate_linear_paths(robot_name, n_paths=n_paths)
    solver = _make_solver(dh, dof)
    timeout_s = timeout_us * 1e-6 if timeout_us > 0 else 0
    rng = np.random.default_rng(12345)

    all_errors = []
    all_devs = []
    timeout_count = 0
    retry_total = 0
    total_start = time.perf_counter()
    for path in paths:
        targets = path["targets"]
        q_cur = path["q_seed"].copy()
        fk_pos = []
        for T in targets:
            t0 = time.perf_counter()
            best_q, best_err = solver.ik(T, q_cur)[:2]
            retries = 0
            # Multi-try: use remaining time budget for retries with perturbed seeds
            if timeout_s > 0 and best_err > 1e-6:
                while (time.perf_counter() - t0) < timeout_s:
                    q_try = q_cur + rng.uniform(-0.5, 0.5, size=dof)
                    q_sol, err, _ = solver.ik(T, q_try)
                    retries += 1
                    if err < best_err:
                        best_q, best_err = q_sol, err
                    if best_err < 1e-6:
                        break
            dt = time.perf_counter() - t0
            retry_total += retries
            if timeout_s > 0 and dt > timeout_s:
                timeout_count += 1
                all_errors.append(float("inf"))
                fk_pos.append(T[:3, 3])
            else:
                all_errors.append(best_err)
                T_fk = forward_kinematics(best_q, dh)
                fk_pos.append(T_fk[:3, 3])
                q_cur = best_q
        lin = compute_linearity(np.array(fk_pos), path["ideal_pos"])
        all_devs.append(lin)
    total_elapsed = time.perf_counter() - total_start

    errors = np.array(all_errors)
    n_total = len(errors)
    success = np.sum(errors < 1e-6)
    total_us = total_elapsed * 1e6
    max_devs = [d["max_dev"] for d in all_devs]
    mean_devs = [d["mean_dev"] for d in all_devs]
    print(f"method: quik_pybind11")
    print(f"num_solves: {n_total}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n_total:.1f}")
    print(f"success_rate: {success/n_total:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"median_error: {np.median(errors):.2e}")
    print(f"timeout_count: {timeout_count}")
    print(f"timeout_rate: {timeout_count/n_total:.4f}")
    print(f"retry_total: {retry_total}")
    print(f"linearity_max_dev: {np.max(max_devs):.2e}")
    print(f"linearity_mean_dev: {np.mean(mean_devs):.2e}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    timeout_us = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    run_benchmark(n, robot_name=robot, mode=mode, timeout_us=timeout_us)
