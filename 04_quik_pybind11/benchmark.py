"""
Benchmark calling the quik IK solver via pybind11.
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
from common.robots import get_robot, generate_test_poses

import quik_binding


def run_benchmark(n=100, robot_name="kuka_kr6", near_init=True):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    _, q_init, targets = generate_test_poses(robot_name, n, near_init=near_init)

    SolverClass = quik_binding.QuikSolver6 if dof == 6 else quik_binding.QuikSolver7
    solver = SolverClass(
        dh, max_iterations=200, max_consecutive_grad_fails=10,
        max_gradient_fails=80, lambda_squared=1e-10, max_linear_step_size=0.34)

    results_err = []
    total_start = time.perf_counter()
    for i in range(n):
        q_star_i, err_i, iter_i = solver.ik(targets[i], q_init[i])
        results_err.append(err_i)
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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    run_benchmark(n, robot_name=robot, near_init=(mode != "random"))
