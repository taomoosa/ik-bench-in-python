"""
Benchmark calling the C++ CLI tool via subprocess for IK.
"""
import subprocess
import struct
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.robots import (get_robot, generate_test_poses, compute_joint_metrics,
                           generate_linear_paths, compute_linearity, forward_kinematics)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _solve_one(cli_path, dh_bin, T, q0, dof):
    T_flat = T.flatten(order="F")
    input_data = dh_bin + struct.pack(f"<16d{dof}d", *T_flat, *q0)
    proc = subprocess.run([cli_path], input=input_data, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI tool failed: {proc.stderr.decode()}")
    output = proc.stdout
    q_star = np.array(struct.unpack(f"<{dof}d", output[:dof*8]))
    error_norm = struct.unpack("<d", output[dof*8:dof*8+8])[0]
    return q_star, error_norm


def run_benchmark(n=100, robot_name="kuka_kr6", mode="near", timeout_us=0):
    if mode == "linear":
        return run_linear_benchmark(n, robot_name, timeout_us=timeout_us)
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    q_true, q_init, targets = generate_test_poses(robot_name, n, mode=mode)

    cli_path = os.path.join(SCRIPT_DIR, "build", f"quik_cli_{dof}dof")
    dh_bin = struct.pack(f"<{dof*4}d", *dh.flatten())

    errors = []
    q_stars = []
    total_start = time.perf_counter()

    for i in range(n):
        q_star, error_norm = _solve_one(cli_path, dh_bin, targets[i], q_init[i], dof)
        errors.append(error_norm)
        q_stars.append(q_star)

    total_elapsed = time.perf_counter() - total_start

    errors = np.array(errors)
    success = np.sum(errors < 1e-6)
    total_us = total_elapsed * 1e6
    print(f"method: quik_subprocess")
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
    cli_path = os.path.join(SCRIPT_DIR, "build", f"quik_cli_{dof}dof")
    dh_bin = struct.pack(f"<{dof*4}d", *dh.flatten())
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
            q_sol, err = _solve_one(cli_path, dh_bin, T, q_cur, dof)
            dt = time.perf_counter() - t0
            if timeout_s > 0 and dt > timeout_s:
                timeout_count += 1
                all_errors.append(float("inf"))
                fk_pos.append(T[:3, 3])
            else:
                all_errors.append(err)
                T_fk = forward_kinematics(q_sol, dh)
                fk_pos.append(T_fk[:3, 3])
                q_cur = q_sol
        lin = compute_linearity(np.array(fk_pos), path["ideal_pos"])
        all_devs.append(lin)
    total_elapsed = time.perf_counter() - total_start

    errors = np.array(all_errors)
    n_total = len(errors)
    success = np.sum(errors < 1e-6)
    total_us = total_elapsed * 1e6
    max_devs = [d["max_dev"] for d in all_devs]
    mean_devs = [d["mean_dev"] for d in all_devs]
    print(f"method: quik_subprocess")
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
