/**
 * quik CLI tool: reads binary data from stdin and writes one IK result to stdout.
 *
 * DOF is set at compile time via the ROBOT_DOF macro.
 *
 * Input (stdin, binary):
 *   ROBOT_DOF * 4 doubles: DH parameters
 *   16 doubles: 4x4 target transform (column-major)
 *   ROBOT_DOF doubles: initial guess q0
 *
 * Output (stdout, binary):
 *   ROBOT_DOF doubles: q_star
 *   1 double: error norm
 *   1 int32: iterations
 */
#ifndef ROBOT_DOF
#define ROBOT_DOF 6
#endif

#include <iostream>
#include <cmath>
#include <cstdio>
#include "Eigen/Dense"
#include "quik/geometry.hpp"
#include "quik/Robot.hpp"
#include "quik/IKSolver.hpp"

using namespace std;
using namespace Eigen;

int main()
{
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

    // Read single pose
    double T_data[16], q0_data[ROBOT_DOF];
    if (fread(T_data, sizeof(double), 16, stdin) != 16) return 1;
    if (fread(q0_data, sizeof(double), ROBOT_DOF, stdin) != ROBOT_DOF) return 1;

    // Setup for single solve (IK expects Dynamic column matrices)
    Matrix<double, Dynamic, 4> Tn(4, 4);
    Map<Matrix4d> T_map(T_data);
    Tn = T_map;

    Matrix<double, ROBOT_DOF, Dynamic> Q0(ROBOT_DOF, 1), Q_star(ROBOT_DOF, 1);
    Matrix<double, 6, Dynamic> E_star(6, 1);
    std::vector<int> iters(1);
    std::vector<quik::BREAKREASON_t> brs(1);

    for (int j = 0; j < ROBOT_DOF; j++)
        Q0(j, 0) = q0_data[j];

    // Solve
    IKS.IK(Tn, Q0, Q_star, E_star, iters, brs);

    // Write result
    for (int j = 0; j < ROBOT_DOF; j++) {
        double val = Q_star(j, 0);
        fwrite(&val, sizeof(double), 1, stdout);
    }
    double err = E_star.col(0).norm();
    fwrite(&err, sizeof(double), 1, stdout);
    int32_t it = iters[0];
    fwrite(&it, sizeof(int32_t), 1, stdout);
    fflush(stdout);

    return 0;
}
