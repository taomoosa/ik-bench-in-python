"""
Inverse kinematics via Newton-Raphson method using Python + NumPy.
Computes FK and Jacobian from DH parameters, then solves IK iteratively.
"""
import numpy as np


def dh_transform(a, alpha, d, theta):
    """Single DH frame transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,   0.0,  1.0],
    ])


def forward_kinematics_all_frames(q, dh):
    """FK for all frames (used for Jacobian computation)."""
    n = len(q)
    frames = [np.eye(4)]
    T = np.eye(4)
    for i in range(n):
        a, alpha, d, theta_offset = dh[i]
        theta = theta_offset + q[i]
        T = T @ dh_transform(a, alpha, d, theta)
        frames.append(T.copy())
    return frames


def geometric_jacobian(q, dh):
    """Compute the geometric Jacobian."""
    frames = forward_kinematics_all_frames(q, dh)
    n = len(q)
    J = np.zeros((6, n))
    o_n = frames[-1][:3, 3]

    for i in range(n):
        z = frames[i][:3, 2]
        o = frames[i][:3, 3]
        J[:3, i] = np.cross(z, o_n - o)
        J[3:, i] = z

    return J


def hgt_diff(T1, T2):
    """Compute twist error between two homogeneous transforms."""
    e = np.zeros(6)
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    d1 = T1[:3, 3]
    d2 = T2[:3, 3]

    e[:3] = d1 - d2

    Re = R1 @ R2.T
    trace = np.trace(Re)
    eps = np.array([
        Re[2, 1] - Re[1, 2],
        Re[0, 2] - Re[2, 0],
        Re[1, 0] - Re[0, 1],
    ])
    eps_norm = np.linalg.norm(eps)

    if trace > -0.99 or eps_norm > 1e-10:
        if eps_norm < 1e-3:
            e[3:] = (0.75 - trace / 12) * eps
        else:
            e[3:] = (np.arctan2(eps_norm, trace - 1) / eps_norm) * eps
    else:
        e[3:] = 1.570796326794897 * (np.diag(Re) + 1)

    return e


def ik_newton(target, q0, dh, max_iter=500, tol=1e-12,
              lambda2=1e-10, max_linear_step=0.34, max_angular_step=1.0):
    """Damped Newton (Levenberg-Marquardt) inverse kinematics."""
    q = q0.copy()
    n = len(q)

    for iteration in range(max_iter):
        frames = forward_kinematics_all_frames(q, dh)
        T_current = frames[-1]

        e = hgt_diff(target, T_current)
        err_norm = np.linalg.norm(e)
        if err_norm < tol:
            return q, err_norm, iteration

        lin_norm = np.linalg.norm(e[:3])
        if lin_norm > max_linear_step:
            e[:3] *= max_linear_step / lin_norm
        ang_norm = np.linalg.norm(e[3:])
        if ang_norm > max_angular_step:
            e[3:] *= max_angular_step / ang_norm

        J = geometric_jacobian(q, dh)

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} e
        JJT = J @ J.T + lambda2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, e)
        q = q + dq

        if np.linalg.norm(dq) < 1e-14:
            break

    return q, np.linalg.norm(hgt_diff(target, forward_kinematics_all_frames(q, dh)[-1])), max_iter
