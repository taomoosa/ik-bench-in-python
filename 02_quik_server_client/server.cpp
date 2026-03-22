/**
 * TCP socket server: runs the quik IK solver internally and
 * responds to requests from a Python client.
 *
 * DOF is set at compile time via the ROBOT_DOF macro.
 * Reads DH parameters from stdin at startup.
 *
 * Protocol (binary):
 *   Request:  16 doubles (4x4 target) + ROBOT_DOF doubles (q0)
 *   Response: ROBOT_DOF doubles (q_star) + 1 double (error) + 1 int32 (iterations)
 *   Shutdown: first 8 bytes are NaN
 */
#ifndef ROBOT_DOF
#define ROBOT_DOF 6
#endif

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include "Eigen/Dense"
#include "quik/geometry.hpp"
#include "quik/Robot.hpp"
#include "quik/IKSolver.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char* argv[])
{
    int port = 9876;
    if (argc > 1) port = atoi(argv[1]);

    // Read DH parameters from stdin
    double dh_data[ROBOT_DOF * 4];
    if (fread(dh_data, sizeof(double), ROBOT_DOF * 4, stdin) != ROBOT_DOF * 4) {
        cerr << "Failed to read DH parameters" << endl;
        return 1;
    }

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

    // Create socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(server_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 1);

    cerr << "Server listening on port " << port << " (DOF=" << ROBOT_DOF << ")" << endl;

    int client_fd = accept(server_fd, nullptr, nullptr);
    setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    cerr << "Client connected" << endl;

    const int REQ_SIZE = (16 + ROBOT_DOF) * sizeof(double);
    const int RESP_SIZE = (ROBOT_DOF + 1) * sizeof(double) + sizeof(int32_t);

    char req_buf[REQ_SIZE];
    char resp_buf[RESP_SIZE];

    while (true) {
        int total_read = 0;
        while (total_read < REQ_SIZE) {
            int n = read(client_fd, req_buf + total_read, REQ_SIZE - total_read);
            if (n <= 0) goto cleanup;
            total_read += n;
        }

        double first_val;
        memcpy(&first_val, req_buf, sizeof(double));
        if (std::isnan(first_val)) break;

        // Parse target transform and initial guess
        Matrix4d Tdes;
        memcpy(Tdes.data(), req_buf, 16 * sizeof(double));

        Vector<double, ROBOT_DOF> q0;
        for (int j = 0; j < ROBOT_DOF; j++) {
            double val;
            memcpy(&val, req_buf + (16 + j) * sizeof(double), sizeof(double));
            q0(j) = val;
        }

        // Solve IK (single pose)
        Matrix<double, Dynamic, 4> Tn(4, 4);
        Tn = Tdes;
        Matrix<double, ROBOT_DOF, Dynamic> Q0(ROBOT_DOF, 1), Q_star(ROBOT_DOF, 1);
        Matrix<double, 6, Dynamic> E_star(6, 1);
        std::vector<int> iters(1);
        std::vector<quik::BREAKREASON_t> brs(1);
        Q0.col(0) = q0;

        IKS.IK(Tn, Q0, Q_star, E_star, iters, brs);

        double error_norm = E_star.col(0).norm();
        int32_t iter_val = iters[0];

        // Send response
        int offset = 0;
        for (int j = 0; j < ROBOT_DOF; j++) {
            double val = Q_star(j, 0);
            memcpy(resp_buf + offset, &val, sizeof(double));
            offset += sizeof(double);
        }
        memcpy(resp_buf + offset, &error_norm, sizeof(double));
        offset += sizeof(double);
        memcpy(resp_buf + offset, &iter_val, sizeof(int32_t));

        int total_written = 0;
        while (total_written < RESP_SIZE) {
            int n = write(client_fd, resp_buf + total_written, RESP_SIZE - total_written);
            if (n <= 0) goto cleanup;
            total_written += n;
        }
    }

cleanup:
    close(client_fd);
    close(server_fd);
    cerr << "Server shutdown" << endl;
    return 0;
}
