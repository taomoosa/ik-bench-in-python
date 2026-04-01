"""
TCP socket client: sends IK requests to the server and runs benchmarks.
"""
import socket
import struct
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.robots import (get_robot, generate_test_poses, compute_joint_metrics,
                           generate_linear_paths, compute_linearity, forward_kinematics)


def _connect(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for _ in range(10):
        try:
            sock.connect((host, port))
            return sock
        except ConnectionRefusedError:
            time.sleep(0.2)
    raise ConnectionError("Cannot connect to server")


def _send_ik(sock, T, q0, dof):
    REQ_SIZE = (16 + dof) * 8
    RESP_SIZE = (dof + 1) * 8 + 4
    T_flat = T.flatten(order="F")
    req = struct.pack(f"<{16}d{dof}d", *T_flat, *q0)
    sock.sendall(req)
    data = b""
    while len(data) < RESP_SIZE:
        chunk = sock.recv(RESP_SIZE - len(data))
        if not chunk:
            raise ConnectionError("Server closed connection")
        data += chunk
    q_star = np.array(struct.unpack(f"<{dof}d", data[:dof*8]))
    error_norm = struct.unpack("<d", data[dof*8:dof*8+8])[0]
    iterations = struct.unpack("<i", data[dof*8+8:dof*8+12])[0]
    return q_star, error_norm, iterations


def _shutdown(sock, dof):
    REQ_SIZE = (16 + dof) * 8
    nan_signal = struct.pack("<d", float("nan")) + b"\x00" * (REQ_SIZE - 8)
    sock.sendall(nan_signal)
    sock.close()


def run_benchmark(n=100, port=19876, host="127.0.0.1", robot_name="kuka_kr6", mode="near", timeout_us=0):
    if mode == "linear":
        return run_linear_benchmark(n, port, host, robot_name, timeout_us=timeout_us)
    robot = get_robot(robot_name)
    dof = robot["dof"]
    q_true, q_init, targets = generate_test_poses(robot_name, n, mode=mode)

    sock = _connect(host, port)

    results = []
    total_start = time.perf_counter()

    for i in range(n):
        q_star, error_norm, iterations = _send_ik(sock, targets[i], q_init[i], dof)
        results.append((q_star, error_norm, iterations))

    total_elapsed = time.perf_counter() - total_start
    _shutdown(sock, dof)

    errors = np.array([r[1] for r in results])
    q_stars = np.array([r[0] for r in results])
    success = np.sum(errors < 1e-6)
    total_us = total_elapsed * 1e6
    print(f"method: quik_server_client")
    print(f"num_solves: {n}")
    print(f"total_time_us: {total_us:.1f}")
    print(f"per_solve_us: {total_us/n:.1f}")
    print(f"success_rate: {success/n:.4f}")
    print(f"max_error: {np.max(errors):.2e}")
    print(f"mean_error: {np.mean(errors):.2e}")
    print(f"median_error: {np.median(errors):.2e}")
    jm = compute_joint_metrics(q_stars, q_true, errors,
                              dh_params=robot["dh_params"], fk_threshold=1e-6)
    print(f"joint_match_rate: {jm['joint_match_rate']:.4f}")
    print(f"joint_err_median: {jm['joint_err_median']:.2e}")
    if "same_aspect_rate" in jm:
        print(f"same_aspect_rate: {jm['same_aspect_rate']:.4f}")
        print(f"diff_aspect_rate: {jm['diff_aspect_rate']:.4f}")
        print(f"cuspidal_swap_rate: {jm['cuspidal_swap_rate']:.4f}")


def run_linear_benchmark(n_paths, port, host, robot_name, timeout_us=0):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    paths = generate_linear_paths(robot_name, n_paths=n_paths)
    timeout_s = timeout_us * 1e-6 if timeout_us > 0 else 0
    rng = np.random.default_rng(12345)

    sock = _connect(host, port)

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
            best_q, best_err, _ = _send_ik(sock, T, q_cur, dof)
            retries = 0
            # Multi-try: use remaining time budget for retries with perturbed seeds
            if timeout_s > 0 and best_err > 1e-6:
                while (time.perf_counter() - t0) < timeout_s:
                    q_try = q_cur + rng.uniform(-0.5, 0.5, size=dof)
                    q_sol, err, _ = _send_ik(sock, T, q_try, dof)
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
    _shutdown(sock, dof)

    errors = np.array(all_errors)
    n_total = len(errors)
    success = np.sum(errors < 1e-6)
    total_us = total_elapsed * 1e6
    max_devs = [d["max_dev"] for d in all_devs]
    mean_devs = [d["mean_dev"] for d in all_devs]
    print(f"method: quik_server_client")
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
    run_benchmark(n, mode=mode, robot_name=robot, timeout_us=timeout_us)
