# ik-bench-in-python

A benchmark project comparing speed, accuracy, and solution-branch behaviour of various inverse kinematics (IK) solvers called from Python.
Benchmarks are run for three robots across eight methods with five initialisation modes.

## Target Robots

| Robot | DOF | Description |
|-------|-----|-------------|
| KUKA KR6 | 6 | Industrial robot with spherical wrist (closed-form solution exists) |
| FANUC CRX-10iA | 6 | Non-spherical wrist with 0.15 m d6 offset (no closed-form solution) |
| Franka Panda | 7 | Redundant collaborative robot (no closed-form solution) |

## Methods

| # | Method | Directory | Description |
|---|--------|-----------|-------------|
| 1 | quik C++ (reference) | `01_quik_cpp_reference/` | Pure C++ quik IK solver (baseline) |
| 2 | quik server-client | `02_quik_server_client/` | C++ TCP server + Python client |
| 3 | quik subprocess | `03_quik_subprocess/` | C++ CLI tool + Python subprocess call |
| 4 | quik pybind11 | `04_quik_pybind11/` | Python call via pybind11 binding |
| 5 | IKPy | `05_ikpy/` | IK using the IKPy Python library |
| 6 | NumPy Newton | `06_numpy_newton/` | Damped Newton method implemented directly in Python+NumPy |
| 7 | TRAC-IK | `07_tracik/` | TRAC-IK solver via pytracik Python binding |
| 8 | OptIK | `08_optik/` | OptIK solver via optik-py Python binding |

## Initialisation Modes

| Mode | Description |
|------|-------------|
| `near` | q_true ± 0.1 — local convergence test |
| `far` | q_true ± 1.0 — medium-distance convergence |
| `j1_offset` | q_true with only J1 offset by ±π |
| `zeros` | All-zero initial guess (home position) |
| `random` | U[−π, π] — global search |
| `linear` | Straight-line Cartesian path — sequential warm-starting linearity test |

### Linear Path Benchmark

When mode is `linear`, the benchmark generates straight-line Cartesian paths
(default: 150 mm, 21 waypoints) starting from random reachable configurations.
The orientation is held constant along each path.

For each path, the first waypoint uses a slightly perturbed seed (q_start ± 0.1).
Subsequent waypoints use the previous IK solution as the initial guess (warm-starting).
After solving all waypoints, FK is computed for each solution and the resulting
end-effector positions are compared to the ideal straight line.

The linearity deviation table reports:
- **Max dev** — worst-case perpendicular distance from the ideal line (mm)
- **Mean dev** — average perpendicular distance across all paths (mm)

This measures how well each solver maintains Cartesian path fidelity when
following a sequential trajectory — important for real-world motion planning.

## Setup

### Prerequisites

- CMake >= 3.14
- g++ (C++17 support)
- Eigen3 (`sudo apt install libeigen3-dev`)
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```bash
# Clone this repository
git clone https://github.com/taomoosa/ik-bench-in-python.git
cd ik-bench-in-python

# Clone the quik library and patch debug output
git clone https://github.com/steffanlloyd/quik.git external/quik

# Install Python dependencies
uv sync

# Build C++ targets (01-03: both 6-DOF and 7-DOF)
for dir in 01_quik_cpp_reference 02_quik_server_client 03_quik_subprocess; do
    mkdir -p $dir/build && cd $dir/build
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
    cd ../..
done

# Build pybind11 module
PYBIND11_DIR=$(uv run python -c "import pybind11; print(pybind11.get_cmake_dir())")
mkdir -p 04_quik_pybind11/build && cd 04_quik_pybind11/build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$PYBIND11_DIR
make -j4
cd ../..
```

If quik is installed at a different location, specify the path via the `QUIK_DIR` CMake variable:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DQUIK_DIR=/path/to/quik
```

> **Note**: quik's `IKSolver.hpp` (around L535) may contain a debug `cout` statement.
> If present, it will corrupt the binary protocol used by methods 01, 02, and 03.
> Comment out the line if needed.

## Running Benchmarks

```bash
# Compare all robots and methods (N=1000, defaults: all robots, modes=near,random)
uv run python benchmark_all.py

# Specify problem count, robots, and modes
uv run python benchmark_all.py 100 kuka_kr6
uv run python benchmark_all.py 500 panda near,far,random
uv run python benchmark_all.py 1000 kuka_kr6,fanuc_crx10ia,panda near,far,j1_offset,zeros,random
uv run python benchmark_all.py 50 kuka_kr6,panda near,linear

# Run individual methods (args: N, mode, robot)
uv run python 04_quik_pybind11/benchmark.py 100 near panda
uv run python 05_ikpy/benchmark.py 100 random kuka_kr6
uv run python 06_numpy_newton/benchmark.py 100 zeros fanuc_crx10ia
uv run python 07_tracik/benchmark.py 100 linear kuka_kr6
uv run python 08_optik/benchmark.py 100 far panda
```

## Benchmark Results

Run the full benchmark to generate current results:

```bash
uv run python benchmark_all.py 1000 kuka_kr6,fanuc_crx10ia,panda
```

### Output Tables

The benchmark outputs the following tables per robot:

1. **Speed** — per-solve time (μs) and ratio to fastest method
2. **Success rate** — fraction of solves with FK error < threshold
3. **Median FK error** — pose-space accuracy
4. **Linearity deviation** (linear mode only) — max/mean perpendicular distance from ideal path (mm)
5. **Joint match rate** — fraction where solved q ≈ true q (2π-wrapped, threshold 0.1 rad)
6. **Aspect match** (6-DOF only) — same-aspect vs cross-aspect solutions
7. **Cuspidal swaps** (6-DOF only) — branch changes without crossing det(J) = 0

### Notes

- All Python-to-C++ calls are issued one at a time, N queries from Python
- subprocess is the slowest due to process startup overhead on every call
- pybind11 maintains near-C++ speed since it is a direct function call
- TRAC-IK and OptIK require URDF files, generated automatically from DH parameters and cached in `common/.urdf_cache/`
- quik IKSolver parameters follow the official sample (`sample_cpp_usage.cpp`)
  - `lambda_squared=1e-10` damping suppresses divergence near singularities

## Project Structure

```
ik-bench-in-python/
├── pyproject.toml              # uv project configuration
├── uv.lock                     # Dependency lock file
├── benchmark_all.py            # Combined benchmark for all methods
├── common/                     # Shared robot definitions and utilities
│   └── robots.py               # DH parameters, FK, Jacobian, URDF generation
├── external/quik/              # quik library (manual clone)
├── 01_quik_cpp_reference/      # Pure C++ benchmark
│   ├── CMakeLists.txt
│   └── main.cpp
├── 02_quik_server_client/      # TCP server-client
│   ├── CMakeLists.txt
│   ├── server.cpp
│   └── client.py
├── 03_quik_subprocess/         # subprocess call
│   ├── CMakeLists.txt
│   ├── quik_cli.cpp
│   └── benchmark.py
├── 04_quik_pybind11/           # pybind11 binding
│   ├── CMakeLists.txt
│   ├── quik_binding.cpp
│   └── benchmark.py
├── 05_ikpy/                    # IKPy benchmark
│   └── benchmark.py
├── 06_numpy_newton/            # NumPy Newton method
│   ├── ik_newton.py
│   └── benchmark.py
├── 07_tracik/                  # TRAC-IK via pytracik
│   └── benchmark.py
└── 08_optik/                   # OptIK via optik-py
    └── benchmark.py
```

## References

- [quik - QuIK inverse kinematics library](https://github.com/steffanlloyd/quik)
- [IKPy - Inverse Kinematics library](https://github.com/Phylliade/ikpy)
- [TRAC-IK - Inverse kinematics solver](https://bitbucket.org/traclabs/trac_ik)
- [OptIK - Rust-based inverse kinematics solver](https://github.com/kylc/optik)
- [QuIK paper (IEEE-TRO)](http://dx.doi.org/10.1109/TRO.2022.3162954)

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

**Exception**: Directories `01_quik_cpp_reference/` through `04_quik_pybind11/`
link against the [quik](https://github.com/steffanlloyd/quik) library, which is
licensed under **AGPL-3.0**. Code in those directories is therefore licensed
under AGPL-3.0. See the `LICENSE` file in each directory for details.

| Directory | License | Reason |
|-----------|---------|--------|
| Root / common / 05-08 | MIT | Permissive dependencies only |
| 01_quik_cpp_reference | AGPL-3.0 | Links against quik (AGPL-3.0) |
| 02_quik_server_client | AGPL-3.0 | Links against quik (AGPL-3.0) |
| 03_quik_subprocess | AGPL-3.0 | Links against quik (AGPL-3.0) |
| 04_quik_pybind11 | AGPL-3.0 | Links against quik (AGPL-3.0) |