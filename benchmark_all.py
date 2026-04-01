"""
Run IK benchmarks for all methods and output comparison tables.
Supports multiple robots (KUKA KR6 6-DOF, FANUC CRX-10iA 6-DOF, Franka Panda 7-DOF).

Usage:
    uv run python benchmark_all.py [N] [robot1,robot2,...] [mode1,mode2,...]
    N: Number of IK problems per method (default: 1000)
       For linear mode, N = number of paths (each with 21 waypoints)
    robots: Comma-separated robot names (default: kuka_kr6,fanuc_crx10ia,panda)
    modes: Comma-separated init modes (default: near,random)
           Available: near, far, j1_offset, zeros, random, linear
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
               else ["kuka_kr6", "fanuc_crx10ia", "panda"])
MODES = (sys.argv[3].split(",") if len(sys.argv) > 3
         else ["near", "random"])
# Per-solve timeout in microseconds for linear mode (0 = no timeout)
TIMEOUT_US = 500


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


def run_python_benchmark(script_path: str, mode: str, robot_name: str,
                         timeout_us: int = 0) -> dict:
    """Run a Python benchmark script."""
    cmd = ["uv", "run", "python", script_path, str(N), mode, robot_name]
    if timeout_us > 0:
        cmd.append(str(timeout_us))
    result = subprocess.run(
        cmd,
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


def run_server_client_benchmark(mode: str, robot_name: str,
                                timeout_us: int = 0) -> dict:
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
        cmd = ["uv", "run", "python", client_path, str(N), mode, robot_name]
        if timeout_us > 0:
            cmd.append(str(timeout_us))
        result = subprocess.run(
            cmd,
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

    is_linear = (mode == "linear")
    timeout_us = TIMEOUT_US if is_linear else 0

    if not is_linear:
        _, q_init, targets = generate_test_poses(robot_name, N, mode=mode)
        cpp_input = pack_test_data(robot_name, targets, q_init)

    # 1. C++ reference (not supported in linear mode — no warm-starting from Python)
    print(f"  [1/8] quik C++ reference ...")
    if is_linear:
        print(f"    SKIPPED (no warm-starting support in pure C++ mode)")
    else:
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
        results["quik_server_client"] = run_server_client_benchmark(
            mode, robot_name, timeout_us=timeout_us)
    else:
        print(f"    SKIPPED (not built: {server_exe})")

    # 3. Subprocess
    print(f"  [3/8] quik subprocess ...")
    cli_exe = os.path.join(
        SCRIPT_DIR, "03_quik_subprocess", "build", f"quik_cli_{dof}dof")
    if os.path.exists(cli_exe):
        results["quik_subprocess"] = run_python_benchmark(
            os.path.join(SCRIPT_DIR, "03_quik_subprocess", "benchmark.py"),
            mode, robot_name, timeout_us=timeout_us)
    else:
        print(f"    SKIPPED (not built: {cli_exe})")

    # 4. pybind11
    print(f"  [4/8] quik pybind11 ...")
    pybind_lib = os.path.join(SCRIPT_DIR, "04_quik_pybind11", "build")
    if os.path.exists(pybind_lib) and any(
            f.startswith("quik_binding") for f in os.listdir(pybind_lib)):
        results["quik_pybind11"] = run_python_benchmark(
            os.path.join(SCRIPT_DIR, "04_quik_pybind11", "benchmark.py"),
            mode, robot_name, timeout_us=timeout_us)
    else:
        print("    SKIPPED (not built)")

    # 5. IKPy
    print(f"  [5/8] IKPy ...")
    results["ikpy"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "05_ikpy", "benchmark.py"), mode, robot_name,
        timeout_us=timeout_us)

    # 6. NumPy Newton
    print(f"  [6/8] NumPy Newton ...")
    results["numpy_newton"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "06_numpy_newton", "benchmark.py"), mode, robot_name,
        timeout_us=timeout_us)

    # 7. TRAC-IK
    print(f"  [7/8] TRAC-IK ...")
    results["tracik"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "07_tracik", "benchmark.py"), mode, robot_name,
        timeout_us=timeout_us)

    # 8. OptIK
    print(f"  [8/8] OptIK ...")
    results["optik"] = run_python_benchmark(
        os.path.join(SCRIPT_DIR, "08_optik", "benchmark.py"), mode, robot_name,
        timeout_us=timeout_us)

    return results


MODE_LABELS = {
    "near":      "Near(±0.1)",
    "far":       "Far(±1.0)",
    "j1_offset": "J1 offset",
    "zeros":     "Zeros",
    "random":    "Random",
    "linear":    f"Linear({TIMEOUT_US}µs)" if TIMEOUT_US > 0 else "Linear path",
}


def print_table(robot_name: str, all_results: dict):
    """Print multi-mode comparison table.

    all_results: {mode_name: {method_name: result_dict}}
    """
    robot = get_robot(robot_name)
    modes = list(all_results.keys())
    mode_labels = [MODE_LABELS.get(m, m) for m in modes]
    n_modes = len(modes)

    # --- Speed table ---
    col_w = 12
    hdr_w = 30 + col_w * n_modes
    print()
    print(f"{'=' * hdr_w}")
    print(f" Speed (µs): {robot['name']} ({robot['dof']}-DOF)  N={N}")
    print(f"{'=' * hdr_w}")
    header = f"{'Method':<30}"
    for ml in mode_labels:
        header += f"{ml:>{col_w}}"
    print(header)
    print(f"{'-' * hdr_w}")

    for name in METHODS:
        row = f"{name:<30}"
        has_any = False
        for mode in modes:
            d = all_results[mode].get(name)
            if d:
                has_any = True
                row += f"{d.get('per_solve_us', 0):>{col_w}.1f}"
            else:
                row += f"{'—':>{col_w}}"
        if has_any:
            print(row)
        else:
            print(f"{name:<30} {'SKIPPED':>{col_w}}")

    # --- Accuracy table (success rate) ---
    print()
    print(f"{'=' * hdr_w}")
    print(f" Success rate: {robot['name']} ({robot['dof']}-DOF)  N={N}")
    print(f"{'=' * hdr_w}")
    header = f"{'Method':<30}"
    for ml in mode_labels:
        header += f"{ml:>{col_w}}"
    print(header)
    print(f"{'-' * hdr_w}")

    for name in METHODS:
        row = f"{name:<30}"
        has_any = False
        for mode in modes:
            d = all_results[mode].get(name)
            if d:
                has_any = True
                row += f"{d.get('success_rate', 0):>{col_w}.4f}"
            else:
                row += f"{'—':>{col_w}}"
        if has_any:
            print(row)
        else:
            print(f"{name:<30} {'SKIPPED':>{col_w}}")

    # --- Accuracy table (median error) ---
    print()
    print(f"{'=' * hdr_w}")
    print(f" Median error: {robot['name']} ({robot['dof']}-DOF)  N={N}")
    print(f"{'=' * hdr_w}")
    header = f"{'Method':<30}"
    for ml in mode_labels:
        header += f"{ml:>{col_w}}"
    print(header)
    print(f"{'-' * hdr_w}")

    for name in METHODS:
        row = f"{name:<30}"
        has_any = False
        for mode in modes:
            d = all_results[mode].get(name)
            if d:
                has_any = True
                v = d.get("median_error", 0)
                row += f"{v:>{col_w}.2e}"
            else:
                row += f"{'—':>{col_w}}"
        if has_any:
            print(row)
        else:
            print(f"{name:<30} {'SKIPPED':>{col_w}}")

    print(f"{'=' * hdr_w}")

    # --- Linearity deviation table (only when 'linear' mode is present) ---
    if "linear" in modes:
        print()
        print(f"{'=' * hdr_w}")
        print(f" Linearity deviation (mm): {robot['name']} ({robot['dof']}-DOF)  N={N} paths")
        print(f"{'=' * hdr_w}")
        lin_header = f"{'Method':<30}{'Max dev':>{col_w}}{'Mean dev':>{col_w}}"
        print(lin_header)
        print(f"{'-' * (30 + col_w * 2)}")
        for name in METHODS:
            d = all_results["linear"].get(name)
            if d and "linearity_max_dev" in d:
                max_mm = d["linearity_max_dev"] * 1000
                mean_mm = d["linearity_mean_dev"] * 1000
                print(f"{name:<30}{max_mm:>{col_w}.2e}{mean_mm:>{col_w}.2e}")
            elif d:
                print(f"{name:<30}{'N/A':>{col_w}}{'N/A':>{col_w}}")
            else:
                print(f"{name:<30}{'—':>{col_w}}{'—':>{col_w}}")
        print(f"{'=' * (30 + col_w * 2)}")

    # --- Timeout rate table (only when 'linear' mode with timeout) ---
    if "linear" in modes and TIMEOUT_US > 0:
        print()
        print(f"{'=' * hdr_w}")
        print(f" Timeout rate (limit={TIMEOUT_US}µs/solve): {robot['name']} ({robot['dof']}-DOF)  N={N} paths")
        print(f"  (quik methods use multi-try with perturbed seeds within budget)")
        print(f"{'=' * hdr_w}")
        to_header = (f"{'Method':<30}{'Timeouts':>{col_w}}{'Rate':>{col_w}}"
                     f"{'Retries':>{col_w}}")
        print(to_header)
        print(f"{'-' * (30 + col_w * 3)}")
        for name in METHODS:
            d = all_results["linear"].get(name)
            if d and "timeout_count" in d:
                tc = int(d["timeout_count"])
                tr = d.get("timeout_rate", 0)
                rt = int(d.get("retry_total", 0))
                rt_str = str(rt) if rt > 0 else "—"
                print(f"{name:<30}{tc:>{col_w}}{tr:>{col_w}.4f}{rt_str:>{col_w}}")
            elif d:
                print(f"{name:<30}{'N/A':>{col_w}}{'N/A':>{col_w}}{'N/A':>{col_w}}")
            else:
                print(f"{name:<30}{'—':>{col_w}}{'—':>{col_w}}{'—':>{col_w}}")
        print(f"{'=' * (30 + col_w * 3)}")

    # --- Joint match rate table ---
    print()
    print(f"{'=' * hdr_w}")
    print(f" Joint match rate (q_solved ≈ q_true): {robot['name']} ({robot['dof']}-DOF)  N={N}")
    print(f"{'=' * hdr_w}")
    header = f"{'Method':<30}"
    for ml in mode_labels:
        header += f"{ml:>{col_w}}"
    print(header)
    print(f"{'-' * hdr_w}")

    for name in METHODS:
        row = f"{name:<30}"
        has_any = False
        for mode in modes:
            d = all_results[mode].get(name)
            if d and "joint_match_rate" in d:
                has_any = True
                row += f"{d['joint_match_rate']:>{col_w}.4f}"
            elif d:
                has_any = True
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        if has_any:
            print(row)
        else:
            print(f"{name:<30} {'SKIPPED':>{col_w}}")

    # --- Joint error median table ---
    print()
    print(f"{'=' * hdr_w}")
    print(f" Joint err median (FK-ok, max|Δq|): {robot['name']} ({robot['dof']}-DOF)  N={N}")
    print(f"{'=' * hdr_w}")
    header = f"{'Method':<30}"
    for ml in mode_labels:
        header += f"{ml:>{col_w}}"
    print(header)
    print(f"{'-' * hdr_w}")

    for name in METHODS:
        row = f"{name:<30}"
        has_any = False
        for mode in modes:
            d = all_results[mode].get(name)
            if d and "joint_err_median" in d:
                has_any = True
                v = d["joint_err_median"]
                row += f"{v:>{col_w}.2e}"
            elif d:
                has_any = True
                row += f"{'N/A':>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        if has_any:
            print(row)
        else:
            print(f"{name:<30} {'SKIPPED':>{col_w}}")

    # --- Selectivity analysis (Method 2) ---
    if "near" in modes and len(modes) > 1:
        print()
        print(f"{'=' * hdr_w}")
        print(f" Selectivity analysis: {robot['name']} ({robot['dof']}-DOF)  N={N}")
        print(f"  (joint_match_rate comparison; near→q_true should ≈1.0)")
        print(f"{'=' * hdr_w}")
        sel_w = hdr_w + col_w
        header = f"{'Method':<30}"
        for ml in mode_labels:
            header += f"{ml:>{col_w}}"
        header += f"{'selectivity':>{col_w}}"
        print(header)
        print(f"{'-' * sel_w}")

        for name in METHODS:
            row = f"{name:<30}"
            near_jm = None
            worst_jm = None
            has_any = False
            for mode in modes:
                d = all_results[mode].get(name)
                if d and "joint_match_rate" in d:
                    has_any = True
                    jm = d["joint_match_rate"]
                    row += f"{jm:>{col_w}.4f}"
                    if mode == "near":
                        near_jm = jm
                    else:
                        if worst_jm is None or jm < worst_jm:
                            worst_jm = jm
                elif d:
                    has_any = True
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{'—':>{col_w}}"
            if near_jm is not None and worst_jm is not None:
                sel = near_jm - worst_jm
                row += f"{sel:>{col_w}.4f}"
            else:
                row += f"{'—':>{col_w}}"
            if has_any:
                print(row)

        print(f"{'-' * sel_w}")
        print(f"  selectivity = near_match - worst_other_match")
        print(f"  High → solver reliably selects the nearest branch from local init")

    # --- Branch classification (det(J) sign, 6-DOF only) ---
    has_branch = any(
        "same_aspect_rate" in (all_results[m].get(name) or {})
        for m in modes for name in METHODS
    )
    if has_branch:
        print()
        print(f"{'=' * hdr_w}")
        print(f" Aspect match (0 path crossings of det(J)=0): {robot['name']}  N={N}")
        print(f"{'=' * hdr_w}")
        header = f"{'Method':<30}"
        for ml in mode_labels:
            header += f"{ml:>{col_w}}"
        print(header)
        print(f"{'-' * hdr_w}")
        for name in METHODS:
            row = f"{name:<30}"
            has_any = False
            for mode in modes:
                d = all_results[mode].get(name)
                if d and "same_aspect_rate" in d:
                    has_any = True
                    row += f"{d['same_aspect_rate']:>{col_w}.4f}"
                elif d:
                    has_any = True
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{'—':>{col_w}}"
            if has_any:
                print(row)

        print()
        print(f"{'=' * hdr_w}")
        print(f" Cuspidal swaps (same aspect, diff branch): {robot['name']}  N={N}")
        print(f"  (FK-ok + 0 path crossings + max|Δq| ≥ 0.1 → cuspidal behavior)")
        print(f"{'=' * hdr_w}")
        header = f"{'Method':<30}"
        for ml in mode_labels:
            header += f"{ml:>{col_w}}"
        print(header)
        print(f"{'-' * hdr_w}")
        for name in METHODS:
            row = f"{name:<30}"
            has_any = False
            for mode in modes:
                d = all_results[mode].get(name)
                if d and "cuspidal_swap_rate" in d:
                    has_any = True
                    row += f"{d['cuspidal_swap_rate']:>{col_w}.4f}"
                elif d:
                    has_any = True
                    row += f"{'N/A':>{col_w}}"
                else:
                    row += f"{'—':>{col_w}}"
            if has_any:
                print(row)

        print(f"{'-' * hdr_w}")
        print(f"  Non-zero cuspidal rate confirms branch changes within the same")
        print(f"  Jacobian aspect (no singular surface crossing on straight-line path).")

    print(f"{'=' * hdr_w}")


def main():
    for robot_name in ROBOT_NAMES:
        robot = get_robot(robot_name)
        print(f"\n{'#' * 70}")
        print(f"# Robot: {robot['name']} ({robot['dof']}-DOF)")
        print(f"# {robot['description']}")
        print(f"{'#' * 70}")

        all_results = {}
        for mode in MODES:
            label = MODE_LABELS.get(mode, mode)
            print(f"\n--- {label} (mode={mode}) ---")
            all_results[mode] = run_all_methods(mode, robot_name)

        print_table(robot_name, all_results)


if __name__ == "__main__":
    main()
