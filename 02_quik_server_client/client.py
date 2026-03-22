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
from common.robots import get_robot, generate_test_poses


def run_benchmark(n=100, port=19876, host="127.0.0.1", robot_name="kuka_kr6", near_init=True):
    robot = get_robot(robot_name)
    dof = robot["dof"]
    _, q_init, targets = generate_test_poses(robot_name, n, near_init=near_init)

    # Connect to server (with retry)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for attempt in range(10):
        try:
            sock.connect((host, port))
            break
        except ConnectionRefusedError:
            time.sleep(0.2)

    REQ_SIZE = (16 + dof) * 8
    RESP_SIZE = (dof + 1) * 8 + 4

    results = []
    total_start = time.perf_counter()

    for i in range(n):
        T_flat = targets[i].flatten(order="F")  # column-major for Eigen
        q0 = q_init[i]
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
        results.append((q_star, error_norm, iterations))

    total_elapsed = time.perf_counter() - total_start

    # Send shutdown signal
    nan_signal = struct.pack("<d", float("nan")) + b"\x00" * (REQ_SIZE - 8)
    sock.sendall(nan_signal)
    sock.close()

    # Report
    errors = np.array([r[1] for r in results])
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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    mode = sys.argv[2] if len(sys.argv) > 2 else "near"
    robot = sys.argv[3] if len(sys.argv) > 3 else "kuka_kr6"
    run_benchmark(n, near_init=(mode != "random"), robot_name=robot)
