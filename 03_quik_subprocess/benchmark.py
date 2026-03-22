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
from common.robots import get_robot, generate_test_poses

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_benchmark(n=100, robot_name="kuka_kr6", near_init=True):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    _, q_init, targets = generate_test_poses(robot_name, n, near_init=near_init)

    cli_path = os.path.join(SCRIPT_DIR, "build", f"quik_cli_{dof}dof")
    dh_bin = struct.pack(f"<{dof*4}d", *dh.flatten())
    RESP_SIZE = (dof + 1) * 8 + 4

    errors = []
    total_start = time.perf_counter()

    for i in range(n):
        T_flat = targets[i].flatten(order="F")
        input_data = dh_bin + struct.pack(f"<16d{dof}d", *T_flat, *q_init[i])

        proc = subprocess.run(
            [cli_path],
            input=input_data,
            capture_output=True,
        )

        if proc.returncode != 0:
            print(f"CLI tool failed (i={i}): {proc.stderr.decode()}", file=sys.stderr)
            sys.exit(1)

        output = proc.stdout
        q_star = np.array(struct.unpack(f"<{dof}d", output[:dof*8]))
        error_norm = struct.unpack("<d", output[dof*8:dof*8+8])[0]
        errors.append(error_norm)

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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    run_benchmark(n, robot_name=robot, near_init=(mode != "random"))
