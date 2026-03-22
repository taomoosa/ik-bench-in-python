"""
Run IK benchmarks for all methods and output comparison tables.
Supports multiple robots (KUKA KR6 6-DOF, Franka Panda 7-DOF).

Usage:
    uv run python benchmark_all.py [N] [robot1,robot2,...]
    N: Number of IK problems per method (default: 1000)
    robots: Comma-separated robot names (default: kuka_kr6,panda)
"""
import subprocess
import struct
import sys
import os
import time
import select
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.robots import get_robot, generate_test_poses

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ROBOT_NAMES = (sys.argv[2].split(",") if len(sys.argv) > 2
               else ["kuka_kr6", "panda"])


def pack_test_data(robot_name, targets, q_init):
    """Pack test data as binary for C++ reference stdin."""
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    n = len(targets)
    data = struct.pack("<i", n)
    # DH parameters (DOF * 4 doubles)
    data += struct.pack(f"<{dof*4}d", *dh.flatten())
    for i in range(n):
        T_flat = targets[i].flatten(order="F")  # column-major for Eigen
        data += struct.pack(f"<16d{dof}d", *T_flat, *q_init[i])
    return data


def parse_output(output: str) -> dict:
    """Parse benchmark output into a dictionary."""
    result = {}
    for line in output.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                if val.lower() in ("inf", "-inf", "nan"):
                    result[key] = float(val)
                elif "." in val or "e" in val.lower():
                    result[key] = float(val)
                else:
                    result[key] = int(val)
            except ValueError:
                result[key] = val
    return result


def run_python_benchmark(script_path: str, mode: str, robot_name: str) -> dict:
    """Run a Python benchmark script."""
    result = subprocess.run(
        ["uv", "run", "python", script_path, str(N), mode, robot_name],
        capture_output=True, text=True, cwd=SCRIPT_DIR,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}", file=sys.stderr)
        return None
    return parse_output(result.stdout)


def run_cpp_benchmark(exe_path: str, input_data: bytes) -> dict:
    """Run a C++ benchmark (pass test data via stdin)."""
    result = subprocess.run(
        [exe_path],
        input=input_data,
        capture_output=True, text=False,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.decode()[:300]}", file=sys.stderr)
        return None
    return parse_output(result.stdout.decode())


def run_server_client_benchmark(mode: str, robot_name: str) -> dict:
    """Run server-client benchmark."""
    robot = get_robot(robot_name)
    dof = robot["dof"]
    dh = robot["dh_params"]
    server_path = os.path.join(
        SCRIPT_DIR, "02_quik_server_client", "build", f"quik_server_{dof}dof")
    client_path = os.path.join(SCRIPT_DIR, "02_quik_server_client", "client.py")

    # DH params to stdin for server startup
    dh_stdin = struct.pack(f"<{dof*4}d", *dh.flatten())

    server_proc = subprocess.Popen(
        [server_path, "19876"],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    server_proc.stdin.write(dh_stdin)
    server_proc.stdin.flush()

    for _ in range(20):
        time.sleep(0.1)
        if server_proc.poll() is not None:
            print(f"  Server exited early: {server_proc.stderr.read().decode()[:200]}",
                  file=sys.stderr)
            return None
        ready, _, _ = select.select([server_proc.stderr], [], [], 0)
        if ready:
            line = server_proc.stderr.readline().decode()
            if "listening" in line.lower():
                break

    try:
        result = subprocess.run(
            ["uv", "run", "python", client_path, str(N), mode, robot_name],
            capture_output=True, text=True, cwd=SCRIPT_DIR,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:300]}", file=sys.stderr)
            return None
        return parse_output(result.stdout)
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)


METHODS = [
    "quik_cpp_reference",
    "quik_server_client",
    "quik_subprocess",
    "quik_pybind11",
    "ikpy",
    "numpy_newton",
    "tracik",
    "optik",
]


def run_all_methods(mode: str, robot_name: str) -> dict:
    """Run all methods for the given robot and mode."""
    robot = get_robot(robot_name)
    dof = robot["dof"]
    results = {}

    near_init = (mode == "near")
    _, q_init, targets = generate_test_poses(robot_name, N, near_init=near_init)
    cpp_input = pack_test_data(robot_name, targets, q_init)

    # 1. C++ reference
    print(f"  [1/8] quik C++ reference ...")
    cpp_exe = os.path.join(
        SCRIPT_DIR, "01_quik_cpp_reference", "build", f"quik_benchmark_{dof}dof")
    if os.path.exists(cpp_exe):
        results["quik_cpp_reference"] = run_cpp_benchmark(cpp_exe, cpp_input)
    else:
        print(f"    SKIPPED (not built: {cpp_exe})")

    # 2. Server-client
    print(f"  [2/8] quik server-client ...")
    server_exe = os.path.join(
        SCRIPT_DIR, "02_quik_server_client", "build", f"quik_server_{dof}dof")
    if os.path.exists(server_exe):
        results["quik_server_client"] = run_server_client_benchmark(mode, robot_name)
    else:
        print(f"    SKIPPED (not built: {server_exe})")

    # 3. Subprocess
    print(f"  [3/8] quik subprocess ...")
    cli_exe = os.path.join(
        SCRIPT_DIR, "03_quik_subprocess", "build", f"quik_cli_{dof}dof")
    if os.path.exists(cli_exe):
        results["quik_subprocess"] = run_python_benchmark(
            os.path.join(SCRIPT_DIR, "03_quik_subprocess", "benchmark.py"),
            mode, robot_name)
    else:
        print(f"    SKIPPED (not built: {cli_exe})")

    # 4. pybind11
    print(f"  [4/8] quik pybind11 ...")
    pybind_lib = os.path.join(SCRIPT_DIR, "04_quik_pybind11", "build")
    if os.path.exists(pybind_lib) and any(
            f.startswith("quik_binding") for f in os.listdir(pybind_lib)):
        results["quik_pybind11"] = run_python_benchmark(
            os.path.join(SCRIPT_DIR, "04_quik_pybind11", "benchmark.py"),
            mode, robot_name)
    else:
        print("    SKIPPED (not built)")

    # 5. IKPy
    print(f"  [5/8] IKPy ...")
    results["ikpy"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "05_ikpy", "benchmark.py"), mode, robot_name)

    # 6. NumPy Newton
    print(f"  [6/8] NumPy Newton ...")
    results["numpy_newton"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "06_numpy_newton", "benchmark.py"), mode, robot_name)

    # 7. TRAC-IK
    print(f"  [7/8] TRAC-IK ...")
    results["tracik"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "07_tracik", "benchmark.py"), mode, robot_name)

    # 8. OptIK
    print(f"  [8/8] OptIK ...")
    results["optik"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "08_optik", "benchmark.py"), mode, robot_name)

    return results


def print_table(robot_name: str, results_near: dict, results_random: dict):
    """Print near vs random comparison table."""
    robot = get_robot(robot_name)
    W = 106

    # Speed comparison
    print()
    print(f"{'=' * W}")
    print(f" Speed: {robot['name']} ({robot['dof']}-DOF) - {robot['description']}  N={N}")
    print(f"{'=' * W}")
    print(f"{'Method':<30} {'Near(us)':>12} {'Random(us)':>14} "
          f"{'Near ratio':>10} {'Random ratio':>12}")
    print(f"{'-' * W}")

    per_near = {n: d.get("per_solve_us", float("inf"))
                for n, d in results_near.items() if d}
    per_rand = {n: d.get("per_solve_us", float("inf"))
                for n, d in results_random.items() if d}
    fastest_near = min(per_near.values()) if per_near else 1.0
    fastest_rand = min(per_rand.values()) if per_rand else 1.0

    for name in METHODS:
        dn = results_near.get(name)
        dr = results_random.get(name)
        if dn is None and dr is None:
            print(f"{name:<30} {'SKIPPED':>12}")
            continue
        pn = dn.get("per_solve_us", 0) if dn else 0
        pr = dr.get("per_solve_us", 0) if dr else 0
        rn = pn / fastest_near if fastest_near > 0 and pn > 0 else 0
        rr = pr / fastest_rand if fastest_rand > 0 and pr > 0 else 0
        print(f"{name:<30} {pn:>12.1f} {pr:>14.1f} {rn:>9.1f}x {rr:>11.1f}x")

    # Accuracy comparison
    print()
    print(f"{'=' * W}")
    print(f" Accuracy: {robot['name']} ({robot['dof']}-DOF)  N={N}")
    print(f"{'=' * W}")
    print(f"{'Method':<30} {'Near succ':>10} {'Random succ':>14} "
          f"{'Near median':>14} {'Random median':>14} {'Near max':>12} {'Random max':>12}")
    print(f"{'-' * W}")

    for name in METHODS:
        dn = results_near.get(name)
        dr = results_random.get(name)
        if dn is None and dr is None:
            print(f"{name:<30} {'SKIPPED':>10}")
            continue
        sn = dn.get("success_rate", 0) if dn else 0
        sr = dr.get("success_rate", 0) if dr else 0
        mn = dn.get("median_error", 0) if dn else 0
        mr = dr.get("median_error", 0) if dr else 0
        xn = dn.get("max_error", 0) if dn else 0
        xr = dr.get("max_error", 0) if dr else 0
        print(f"{name:<30} {sn:>10.4f} {sr:>14.4f} "
              f"{mn:>14.2e} {mr:>14.2e} {xn:>12.2e} {xr:>12.2e}")

    print(f"{'=' * W}")


def main():
    for robot_name in ROBOT_NAMES:
        robot = get_robot(robot_name)
        print(f"\n{'#' * 70}")
        print(f"# Robot: {robot['name']} ({robot['dof']}-DOF)")
        print(f"# {robot['description']}")
        print(f"{'#' * 70}")

        print(f"\n--- Near initial guess (q0 = q_true +/- 0.1) ---")
        results_near = run_all_methods("near", robot_name)

        print(f"\n--- Random initial guess (q0 ~ U[-pi, pi]) ---")
        results_random = run_all_methods("random", robot_name)

        print_table(robot_name, results_near, results_random)


if __name__ == "__main__":
    main()
