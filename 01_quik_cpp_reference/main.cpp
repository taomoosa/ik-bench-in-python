/**
 * Inverse kinematics benchmark using pure C++ quik.
 *
 * DOF is set at compile time via the ROBOT_DOF macro.
 * Reads DH parameters and test data from stdin, runs batch IK.
 *
 * Input (stdin, binary):
 *   int32: N (number of problems)
 *   ROBOT_DOF * 4 doubles: DH parameters (a, alpha, d, theta per row)
 *   N times:
 *     16 doubles: 4x4 target transform (column-major)
 *     ROBOT_DOF doubles: initial guess q0
 */
#ifndef ROBOT_DOF
#define ROBOT_DOF 6
#endif

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "Eigen/Dense"
#include "quik/geometry.hpp"
#include "quik/Robot.hpp"
#include "quik/IKSolver.hpp"

using namespace std;
using namespace Eigen;

int main()
{
    // Read N
    int32_t N;
    if (fread(&N, sizeof(int32_t), 1, stdin) != 1) return 1;

    // Read DH parameters
    double dh_data[ROBOT_DOF * 4];
    if (fread(dh_data, sizeof(double), ROBOT_DOF * 4, stdin) != ROBOT_DOF * 4) return 1;

    Matrix<double, ROBOT_DOF, 4> DH;
    for (int i = 0; i < ROBOT_DOF; i++)
        for (int j = 0; j < 4; j++)
            DH(i, j) = dh_data[i * 4 + j];

    Vector<quik::JOINTTYPE_t, ROBOT_DOF> joint_types;
    Vector<double, ROBOT_DOF> joint_signs;
    for (int i = 0; i < ROBOT_DOF; i++) {
        joint_types(i) = quik::JOINT_REVOLUTE;
        joint_signs(i) = 1.0;
    }

    auto R = std::make_shared<quik::Robot<ROBOT_DOF>>(
        DH, joint_types, joint_signs,
        Matrix4d::Identity(), Matrix4d::Identity()
    );
    const quik::IKSolver<ROBOT_DOF> IKS(
        R, 200, quik::ALGORITHM_QUIK, 1e-12, 1e-14,
        0.05, 10, 80, 1e-10, 0.34, 1.0);

    // Read test data
    Matrix<double, ROBOT_DOF, Dynamic> Q0(ROBOT_DOF, N), Q_star(ROBOT_DOF, N);
    Matrix<double, 6, Dynamic> e_star(6, N);
    std::vector<int> iter(N);
    std::vector<quik::BREAKREASON_t> breakReason(N);
    Matrix<double, Dynamic, 4> Tn(N * 4, 4);

    for (int i = 0; i < N; i++) {
        double T_data[16];
        double q0_data[ROBOT_DOF];
        if (fread(T_data, sizeof(double), 16, stdin) != 16) return 1;
        if (fread(q0_data, sizeof(double), ROBOT_DOF, stdin) != ROBOT_DOF) return 1;
        Map<Matrix4d> T_map(T_data);
        Tn.middleRows<4>(i * 4) = T_map;
        for (int j = 0; j < ROBOT_DOF; j++)
            Q0(j, i) = q0_data[j];
    }

    // Benchmark IK (batch)
    auto startTime = chrono::high_resolution_clock::now();
    IKS.IK(Tn, Q0, Q_star, e_star, iter, breakReason);
    chrono::duration<double, std::micro> elapsed = chrono::high_resolution_clock::now() - startTime;

    // Collect error norms
    std::vector<double> errors(N);
    int success_count = 0;
    double max_error = 0, sum_error = 0;
    for (int i = 0; i < N; i++) {
        double err = e_star.col(i).norm();
        errors[i] = err;
        sum_error += err;
        if (err < 1e-6) success_count++;
        if (err > max_error) max_error = err;
    }
    std::sort(errors.begin(), errors.end());
    double median_error = (N % 2 == 0) ?
        (errors[N / 2 - 1] + errors[N / 2]) / 2.0 : errors[N / 2];

    cout << "method: quik_cpp_reference" << endl;
    cout << "num_solves: " << N << endl;
    cout << "total_time_us: " << elapsed.count() << endl;
    cout << "per_solve_us: " << elapsed.count() / N << endl;
    cout << "success_rate: " << (double)success_count / N << endl;
    cout << "max_error: " << max_error << endl;
    cout << "mean_error: " << sum_error / N << endl;
    cout << "median_error: " << median_error << endl;

    return 0;
}
