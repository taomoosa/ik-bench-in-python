/**
 * pybind11 binding: exposes quik Robot + IKSolver for direct Python calls.
 * Provides both DOF=6 (QuikSolver6) and DOF=7 (QuikSolver7).
 * All IKSolver parameters are also exposed.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "quik/geometry.hpp"
#include "quik/Robot.hpp"
#include "quik/IKSolver.hpp"

namespace py = pybind11;
using namespace Eigen;

template<int DOF>
class QuikSolverImpl {
public:
    std::shared_ptr<quik::Robot<DOF>> robot;
    std::unique_ptr<quik::IKSolver<DOF>> solver;

    QuikSolverImpl(py::array_t<double> dh_params,
                   int max_iterations = 100,
                   int algorithm = 0,
                   double exit_tolerance = 1e-12,
                   double minimum_step_size = 1e-14,
                   double relative_improvement_tolerance = 0.05,
                   int max_consecutive_grad_fails = 5,
                   int max_gradient_fails = 20,
                   double lambda_squared = 0.0,
                   double max_linear_step_size = -1.0,
                   double max_angular_step_size = 1.0)
    {
        auto buf = dh_params.unchecked<2>();
        Matrix<double, DOF, 4> DH;
        for (int i = 0; i < DOF; i++)
            for (int j = 0; j < 4; j++)
                DH(i, j) = buf(i, j);

        Vector<quik::JOINTTYPE_t, DOF> joint_types;
        Vector<double, DOF> joint_signs;
        for (int i = 0; i < DOF; i++) {
            joint_types(i) = quik::JOINT_REVOLUTE;
            joint_signs(i) = 1.0;
        }

        robot = std::make_shared<quik::Robot<DOF>>(
            DH, joint_types, joint_signs,
            Matrix4d::Identity(), Matrix4d::Identity()
        );
        solver = std::make_unique<quik::IKSolver<DOF>>(
            robot,
            max_iterations,
            static_cast<quik::ALGORITHM_t>(algorithm),
            exit_tolerance,
            minimum_step_size,
            relative_improvement_tolerance,
            max_consecutive_grad_fails,
            max_gradient_fails,
            lambda_squared,
            max_linear_step_size,
            max_angular_step_size
        );
    }

    py::array_t<double> fk(py::array_t<double> q_arr) {
        auto q_buf = q_arr.unchecked<1>();
        Vector<double, DOF> q;
        for (int i = 0; i < DOF; i++) q(i) = q_buf(i);

        Matrix4d T;
        robot->FKn(q, T);

        auto result = py::array_t<double>({4, 4});
        auto r = result.mutable_unchecked<2>();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                r(i, j) = T(i, j);
        return result;
    }

    py::tuple ik(py::array_t<double> target, py::array_t<double> q0_arr) {
        auto t_buf = target.unchecked<2>();
        auto q0_buf = q0_arr.unchecked<1>();

        Matrix<double, Dynamic, 4> Tn(4, 4);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                Tn(r, c) = t_buf(r, c);

        Matrix<double, DOF, Dynamic> Q0(DOF, 1), Q_star(DOF, 1);
        Matrix<double, 6, Dynamic> E_star(6, 1);
        std::vector<int> iters(1);
        std::vector<quik::BREAKREASON_t> brs(1);

        for (int j = 0; j < DOF; j++) Q0(j, 0) = q0_buf(j);

        solver->IK(Tn, Q0, Q_star, E_star, iters, brs);

        auto q_result = py::array_t<double>(DOF);
        auto qr = q_result.mutable_unchecked<1>();
        for (int j = 0; j < DOF; j++) qr(j) = Q_star(j, 0);

        return py::make_tuple(q_result, E_star.col(0).norm(), iters[0]);
    }
};

template<int DOF>
void register_solver(py::module& m, const char* name) {
    py::class_<QuikSolverImpl<DOF>>(m, name)
        .def(py::init<py::array_t<double>, int, int, double, double, double, int, int, double, double, double>(),
             py::arg("dh_params"),
             py::arg("max_iterations") = 100,
             py::arg("algorithm") = 0,
             py::arg("exit_tolerance") = 1e-12,
             py::arg("minimum_step_size") = 1e-14,
             py::arg("relative_improvement_tolerance") = 0.05,
             py::arg("max_consecutive_grad_fails") = 5,
             py::arg("max_gradient_fails") = 20,
             py::arg("lambda_squared") = 0.0,
             py::arg("max_linear_step_size") = -1.0,
             py::arg("max_angular_step_size") = 1.0)
        .def("fk", &QuikSolverImpl<DOF>::fk, py::arg("q"))
        .def("ik", &QuikSolverImpl<DOF>::ik, py::arg("target"), py::arg("q0"));
}

PYBIND11_MODULE(quik_binding, m) {
    m.doc() = "QuIK inverse kinematics solver - pybind11 binding";
    m.attr("ALGORITHM_QUIK") = 0;
    m.attr("ALGORITHM_NR") = 1;
    m.attr("ALGORITHM_BFGS") = 2;
    register_solver<6>(m, "QuikSolver6");
    register_solver<7>(m, "QuikSolver7");
}
