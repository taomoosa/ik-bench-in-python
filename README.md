# quik-from-python

A benchmark project comparing speed and accuracy of various methods for calling robot inverse kinematics (IK) from Python.
Benchmarks are run for two robots across eight methods.

## Target Robots

| Robot | DOF | Description |
|-------|-----|-------------|
| KUKA KR6 | 6 | Industrial robot with spherical wrist (closed-form solution exists) |
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

## Setup

### Prerequisites

- CMake >= 3.14
- g++ (C++17 support)
- Eigen3 (`sudo apt install libeigen3-dev`)
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```bash
# Clone this repository
git clone <repository-url>
cd quik-from-python

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
# Compare all robots and methods (N=1000, default)
uv run python benchmark_all.py

# Specify problem count and robot
uv run python benchmark_all.py 100 kuka_kr6
uv run python benchmark_all.py 500 panda
uv run python benchmark_all.py 1000 kuka_kr6,panda

# Run individual methods (args: N, mode[near/random], robot)
uv run python 04_quik_pybind11/benchmark.py 100 near panda
uv run python 05_ikpy/benchmark.py 100 random kuka_kr6
uv run python 06_numpy_newton/benchmark.py 100 near panda
uv run python 07_tracik/benchmark.py 100 near kuka_kr6
uv run python 08_optik/benchmark.py 100 random panda
```

## Benchmark Results

Run the full benchmark to generate current results:

```bash
uv run python benchmark_all.py 1000 kuka_kr6,panda
```

### Notes

- All Python-to-C++ calls are issued one at a time, N queries from Python
- subprocess is the slowest due to process startup overhead on every call
- pybind11 maintains near-C++ speed since it is a direct function call
- TRAC-IK and OptIK require URDF files, generated automatically from DH parameters
- quik IKSolver parameters follow the official sample (`sample_cpp_usage.cpp`)
  - `lambda_squared=1e-10` damping suppresses divergence near singularities

## Project Structure

```
quik-from-python/
├── pyproject.toml              # uv project configuration
├── uv.lock                     # Dependency lock file
├── benchmark_all.py            # Combined benchmark for all methods
├── common/                     # Shared robot definitions
│   └── robots.py               # DH parameters (KR6, Panda)
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