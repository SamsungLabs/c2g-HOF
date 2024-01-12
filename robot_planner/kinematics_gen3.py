"""
Kinematics of Gen3
"""

import numpy as np
import math


def position6D(th):
    p = np.zeros(6)
    p[0] = th[0][3]
    p[1] = th[1][3]
    p[2] = th[2][3]
    p[3] = math.atan2(th[2][1], th[2][2])
    if th[2][0] > 0.99999999:
        th[2][0] = 1.0
    if th[2][0] < -0.99999999:
        th[2][0] = -1.0

    p[4] = -math.asin(th[2][0])
    p[5] = math.atan2(th[1][0], th[0][0])
    return p


def Transform_rotateX(t, a):
    ca = math.cos(a)
    sa = math.sin(a)

    for i in range(3):
        ty = t[i][1].copy()
        tz = t[i][2].copy()
        t[i][1] = ca * ty + sa * tz
        t[i][2] = -sa * ty + ca * tz

    return t


def Transform_rotateY(ta, a):
    ca = math.cos(a)
    sa = math.sin(a)

    for i in range(3):
        tx = ta[i][0].copy()
        tz = ta[i][2].copy()
        ta[i][0] = ca * tx - sa * tz
        ta[i][2] = sa * tx + ca * tz
    return ta


def Transform_rotateZ(t, a):
    ca = math.cos(a)
    sa = math.sin(a)

    for i in range(3):
        tx = t[i][0].copy()
        ty = t[i][1].copy()
        t[i][0] = ca * tx + sa * ty
        t[i][1] = -sa * tx + ca * ty
    return t


def Transform_translateX(t, x):
    t[0][3] = t[0][3] + t[0][0] * x
    t[1][3] = t[1][3] + t[1][0] * x
    t[2][3] = t[2][3] + t[2][0] * x
    return t


def Transform_translateY(t, y):

    t[0][3] = t[0][3] + t[0][1] * y
    t[1][3] = t[1][3] + t[1][1] * y
    t[2][3] = t[2][3] + t[2][1] * y
    return t


def Transform_translateZ(t, z):

    t[0][3] = t[0][3] + t[0][2] * z
    t[1][3] = t[1][3] + t[1][2] * z
    t[2][3] = t[2][3] + t[2][2] * z
    return t


def Transform_mDH(t, alpha, a, theta, d):

    t = Transform_translateX(t, a)
    t = Transform_rotateX(t, alpha)
    t = Transform_translateZ(t, d)
    t = Transform_rotateZ(t, theta)

    return t


def forward_kinematics(q, alpha=1.0):

    output_pt = np.zeros((4, 6))
    M_PI = np.pi
    len1 = 0.15643
    len2 = 0.12838
    len3 = 0.21038
    len4 = 0.21038
    len5 = 0.20843
    len6 = 0.10593
    len7 = 0.10593
    len8 = 0.061525 + 0.12

    offset1 = 0.005375
    offset2 = 0.006375
    offset3 = 0.006375
    offset4 = 0.006375

    t = np.eye(4)
    t = Transform_mDH(t, M_PI, 0, q[0], -len1)

    t = Transform_translateY(t, offset1)
    t = Transform_translateZ(t, -len2)
    t = Transform_mDH(t, M_PI / 2.0, 0, q[1], 0)
    pt1 = position6D(t)

    t = Transform_translateY(t, -len3)
    t = Transform_translateZ(t, -offset2)
    t = Transform_mDH(t, -M_PI / 2.0, 0, q[2], 0)

    t = Transform_translateY(t, offset3)
    t = Transform_translateZ(t, -len4)
    t = Transform_mDH(t, M_PI / 2.0, 0, q[3], 0)
    pt2 = position6D(t)

    t = Transform_translateY(t, -offset4)
    t = Transform_mDH(t, -M_PI / 2.0, 0, q[4], -len5)

    t = Transform_mDH(t, M_PI / 2.0, 0, 0, 0)
    t = Transform_translateY(t, -len6)
    t = Transform_mDH(t, 0, 0, q[5], 0)
    pt3 = position6D(t)

    t = Transform_mDH(t, -M_PI / 2.0, 0, q[6], -len7)
    t = Transform_mDH(t, M_PI, 0, 0, len8)
    t = Transform_rotateZ(t, M_PI / 2)
    pt4 = position6D(t)

    output_pt[0, :] = pt1
    output_pt[1, :] = pt2
    output_pt[2, :] = pt3
    output_pt[3, :] = pt4

    return output_pt
