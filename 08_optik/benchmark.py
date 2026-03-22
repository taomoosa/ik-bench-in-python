"""
IK benchmark using OptIK (optik-py).
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.robots import get_robot, generate_test_poses, get_urdf_path


def run_benchmark(n=100, robot_name="kuka_kr6", near_init=True):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    urdf_path = get_urdf_path(robot_name)

    import optik
    optik_robot = optik.Robot.from_urdf_file(urdf_path, "base_link", "ee_link")
    # Single-threaded for fair per-solve timing comparison
    optik_robot.set_parallelism(1)
    config = optik.SolverConfig(solution_mode='speed', max_time=0.01,
                                tol_f=1e-10)

    q_true, q_init, targets = generate_test_poses(
        robot_name, n, near_init=near_init)

    errors = []
    total_start = time.perf_counter()

    for i in range(n):
        T = targets[i]
        T_list = T.tolist()
        q_sol, residual = optik_robot.ik(config, T_list, q_init[i].tolist())
        if q_sol is not None:
            from common.robots import forward_kinematics
            T_sol = forward_kinematics(np.array(q_sol), robot["dh_params"])
            err = np.linalg.norm(T_sol[:3, :] - T[:3, :])
            errors.append(err)
        else:
            errors.append(float("inf"))

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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    run_benchmark(n, robot_name=robot, near_init=(mode != "random"))
