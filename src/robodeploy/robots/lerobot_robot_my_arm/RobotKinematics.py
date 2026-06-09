import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from pynput import keyboard
import math
import os

# DH 参数类，存储 alpha, theta, a, d 的值
class DHParam:
    def __init__(self, alpha, theta, a, d):
        self.alpha = alpha
        self.theta = theta
        self.a = a
        self.d = d

class RobotKinematics:
    def __init__(self):
        self.curDH = [
                DHParam(    0,          0,                  0,          0.1205  ),
                DHParam(    np.pi/2,    np.pi,              0,          0       ),
                DHParam(    0,          -166.26/180*np.pi,  0.37468,    0       ),
                DHParam(    0,          -13.74/180*np.pi,   0.28486,    0       ),
                DHParam(    -np.pi/2,   -np.pi/2,           0.0784,     0       ),
                DHParam(    -np.pi/2,   0,                  0,          0.206   ),
            ]

        self.joint_range = [
                    [   -np.pi,     np.pi   ],
                    [   -np.pi,     0       ],
                    [   0,          np.pi   ],
                    [   -1.8,       1.4     ],
                    [   -2,         2       ],
                    [   -np.pi,     np.pi   ],
                    [   0,          0.05    ],  #   Gripper
        ]

    def ForwardKinematics(self, theta_input=None):
        # 初始变换矩阵（单位矩阵）
        transform = np.eye(4)

        # 遍历所有的 DH 参数，计算每一关节的变换
        for i in range(len(self.curDH)):
            theta = theta_input[i] + self.curDH[i].theta if theta_input is not None else self.curDH[i].theta
            T = self.DHToMatrix(self.curDH[i].alpha, theta, self.curDH[i].a, self.curDH[i].d)
            transform = np.dot(transform, T)

        return transform

    # 从变换矩阵提取位置和姿态，返回6维数组 [x, y, z, roll, pitch, yaw]
    def extract_pose(self, transform):
        pose = np.zeros(6)

        # 提取位置信息
        pose[0] = transform[0, 3]  # X
        pose[1] = transform[1, 3]  # Y
        pose[2] = transform[2, 3]  # Z

        # 提取姿态信息 (ZYX欧拉角)
        sy = math.sqrt(transform[0, 0] * transform[0, 0] + transform[1, 0] * transform[1, 0])

        singular = sy < 1e-6  # 奇异情况判断

        if not singular:
            pose[3] = math.atan2(transform[2, 1], transform[2, 2])  # Roll
            pose[4] = math.atan2(-transform[2, 0], sy)              # Pitch
            pose[5] = math.atan2(transform[1, 0], transform[0, 0])  # Yaw
        else:
            pose[3] = 0
            pose[4] = math.atan2(-transform[2, 0], sy)
            pose[5] = math.atan2(-transform[1, 2], transform[1, 1])

        return pose
    
    def set_boundary(self, joint_range, input_q, idx):
        input_q = min(max(input_q, joint_range[idx][0]), joint_range[idx][1])
        return input_q

    # DHToMatrix 函数
    def DHToMatrix(self, alpha, theta, a, d):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        ct = np.cos(theta)
        st = np.sin(theta)

        # DH 变换矩阵
        matrix = np.array([
            [   ct,    -st,    0,      a       ],
            [   st*ca, ct*ca,  -sa,    -d*sa   ],
            [   st*sa, ct*sa,  ca,     d*ca    ],
            [   0,     0,      0,      1       ]
        ])
        return matrix
    
    # 角度转换为弧度
    def degrees_to_radians(self, degrees):
        return degrees * math.pi / 180

    # 创建绕X轴的旋转矩阵
    def create_rotation_matrix_x(self, alpha):
        return np.array([
            [1, 0, 0],
            [0, math.cos(alpha), -math.sin(alpha)],
            [0, math.sin(alpha), math.cos(alpha)]
        ])

    # 创建绕Y轴的旋转矩阵
    def create_rotation_matrix_y(self, beta):
        return np.array([
            [math.cos(beta), 0, math.sin(beta)],
            [0, 1, 0],
            [-math.sin(beta), 0, math.cos(beta)]
        ])

    # 创建绕Z轴的旋转矩阵
    def create_rotation_matrix_z(self, gamma):
        return np.array([
            [math.cos(gamma), -math.sin(gamma), 0],
            [math.sin(gamma), math.cos(gamma), 0],
            [0, 0, 1]
        ])

    # 创建末端姿态矩阵
    def create_end_effector_pose(self, R, xd, yd, zd):
        return np.array([
            [R[0, 0], R[0, 1], R[0, 2], xd],
            [R[1, 0], R[1, 1], R[1, 2], yd],
            [R[2, 0], R[2, 1], R[2, 2], zd],
            [0, 0, 0, 1]
        ])

    # 把末端位置和姿态转换为4x4变换矩阵
    def p2r_matrix(self, x1, x2, x3, x4, x5, x6):
        # alpha = degrees_to_radians(x4)  # 绕X轴旋转
        # beta = degrees_to_radians(x5)   # 绕Y轴旋转
        # gamma = degrees_to_radians(x6)  # 绕Z轴旋转

        # 默认输入是弧度制
        alpha = x4
        beta = x5
        gamma = x6

        # 假设的末端位置坐标
        xd = x1
        yd = x2
        zd = x3

        # 计算各个轴的旋转矩阵
        Rx = self.create_rotation_matrix_x(alpha)
        Ry = self.create_rotation_matrix_y(beta)
        Rz = self.create_rotation_matrix_z(gamma)

        # 计算总旋转矩阵（注意顺序：先Z，再Y，最后X）
        R = np.dot(np.dot(Rz, Ry), Rx)
        A = self.create_end_effector_pose(R, xd, yd, zd)

        return A

    def check_range(self, joint_range, q):
        ret = True
        for i in range(len(q)):
            if q[i] - joint_range[i][1] > 0.001 or joint_range[i][0] - q[i] > 0.001:
                print("joint %d out of range" % (i + 1))
                ret = False
                break
        return ret

    def NormalizeJointAngles(self, jointAngles):
        res = [0] * len(jointAngles)
        for i in range(len(jointAngles)):
            angle = jointAngles[i]
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle <= -math.pi:
                angle += 2 * math.pi
            res[i] = angle
        return res

    def InverseKinematics(self, matrix, q):
        nx, ox, ax, px = matrix[0]
        ny, oy, ay, py = matrix[1]
        nz, oz, az, pz = matrix[2]

        d1 = self.curDH[0].d
        d2 = self.curDH[1].d
        d3 = self.curDH[2].d
        d4 = self.curDH[3].d
        d5 = self.curDH[4].d
        d6 = self.curDH[5].d

        a2 = self.curDH[2].a
        a3 = self.curDH[3].a
        a4 = self.curDH[4].a

        # ---- q1 ----
        m1 = d6 * ax - px
        n1 = d6 * ay - py

        q1_1 = math.atan2(n1, m1)
        q1_2 = q1_1 - math.pi if q1_1 >= 0 else q1_1 + math.pi

        # ---- q5 (对应 q1_1) ----
        sinq5_1 = ay * math.cos(q1_1) - ax * math.sin(q1_1)
        if abs(sinq5_1) <= 1:
            q5_1 = math.asin(sinq5_1)
            q5_2 = math.pi - q5_1 if q5_1 >= 0 else -math.pi - q5_1
        else:
            q5_1 = float('nan')
            q5_2 = float('nan')

        # ---- q5 (q1_2对应) ----
        sinq5_2 = ay * math.cos(q1_2) - ax * math.sin(q1_2)
        if abs(sinq5_2) <= 1:
            q5_3 = math.asin(sinq5_2)
            q5_4 = math.pi - q5_3 if q5_3 >= 0 else -math.pi - q5_3
        else:
            q5_3 = float('nan')
            q5_4 = float('nan')

        # ---- q6 (q1_1对应) ----
        u6_1 = ox * math.sin(q1_1) - oy * math.cos(q1_1)
        v6_1 = ny * math.cos(q1_1) - nx * math.sin(q1_1)

        if abs(sinq5_1) <= 1:
            q6_1 = math.atan2((-math.cos(q5_1)) / math.sqrt(u6_1**2 + v6_1**2), 0) - math.atan2(v6_1, u6_1)
            q6_2 = math.atan2((-math.cos(q5_2)) / math.sqrt(u6_1**2 + v6_1**2), 0) - math.atan2(v6_1, u6_1)
        else:
            q6_1 = float('nan')
            q6_2 = float('nan')

        # ---- q6 (q1_2对应) ----
        u6_2 = ox * math.sin(q1_2) - oy * math.cos(q1_2)
        v6_2 = ny * math.cos(q1_2) - nx * math.sin(q1_2)

        if abs(sinq5_2) <= 1:
            q6_3 = math.atan2((-math.cos(q5_3)) / math.sqrt(u6_2**2 + v6_2**2), 0) - math.atan2(v6_2, u6_2)
            q6_4 = math.atan2((-math.cos(q5_4)) / math.sqrt(u6_2**2 + v6_2**2), 0) - math.atan2(v6_2, u6_2)
        else:
            q6_3 = float('nan')
            q6_4 = float('nan')

        # ---- q3 (这里只写 q1_1 对应前两个) ----

        t14_1 = (px - ax * d6 - a4 * (ax * math.cos(q5_1) + nx * math.sin(q5_1) * math.cos(q6_1) - ox * math.sin(q5_1) * math.sin(q6_1))) * math.cos(q1_1) \
            + (py - ay * d6 - a4 * (ay * math.cos(q5_1) + ny * math.sin(q5_1) * math.cos(q6_1) - oy * math.sin(q5_1) * math.sin(q6_1))) * math.sin(q1_1)

        t34_1 = -a4 * az * math.cos(q5_1) \
            - a4 * nz * math.sin(q5_1) * math.cos(q6_1) \
            + a4 * oz * math.sin(q5_1) * math.sin(q6_1) \
            - az * d6 - d1 + pz

        t14_2 = (px - ax * d6 - a4 * (ax * math.cos(q5_2) + nx * math.sin(q5_2) * math.cos(q6_2) - ox * math.sin(q5_2) * math.sin(q6_2))) * math.cos(q1_1) \
            + (py - ay * d6 - a4 * (ay * math.cos(q5_2) + ny * math.sin(q5_2) * math.cos(q6_2) - oy * math.sin(q5_2) * math.sin(q6_2))) * math.sin(q1_1)

        t34_2 = -a4 * az * math.cos(q5_2) \
            - a4 * nz * math.sin(q5_2) * math.cos(q6_2) \
            + a4 * oz * math.sin(q5_2) * math.sin(q6_2) \
            - az * d6 - d1 + pz

        cosq3_1 = (t14_1**2 + t34_1**2 - a2**2 - a3**2) / (2 * a2 * a3)
        cosq3_2 = (t14_2**2 + t34_2**2 - a2**2 - a3**2) / (2 * a2 * a3)

        if abs(sinq5_1) <= 1:
            if abs(cosq3_1) <= 1:
                q3_1 = math.acos(cosq3_1) - self.curDH[2].theta
                q3_2 = -math.acos(cosq3_1) - self.curDH[2].theta
            else:
                q3_1 = float('nan')
                q3_2 = float('nan')

            if abs(cosq3_2) <= 1:
                q3_3 = math.acos(cosq3_2) - self.curDH[2].theta
                q3_4 = -math.acos(cosq3_2) - self.curDH[2].theta
            else:
                q3_3 = float('nan')
                q3_4 = float('nan')
        else:
            q3_1 = q3_2 = q3_3 = q3_4 = float('nan')

        t14_3 = (px - ax * d6 - a4 * (ax * math.cos(q5_3) + nx * math.sin(q5_3) * math.cos(q6_3) - ox * math.sin(q5_3) * math.sin(q6_3))) * math.cos(q1_2) \
            + (py - ay * d6 - a4 * (ay * math.cos(q5_3) + ny * math.sin(q5_3) * math.cos(q6_3) - oy * math.sin(q5_3) * math.sin(q6_3))) * math.sin(q1_2)

        t34_3 = -a4 * az * math.cos(q5_3) \
            - a4 * nz * math.sin(q5_3) * math.cos(q6_3) \
            + a4 * oz * math.sin(q5_3) * math.sin(q6_3) \
            - az * d6 - d1 + pz

        t14_4 = (px - ax * d6 - a4 * (ax * math.cos(q5_4) + nx * math.sin(q5_4) * math.cos(q6_4) - ox * math.sin(q5_4) * math.sin(q6_4))) * math.cos(q1_2) \
            + (py - ay * d6 - a4 * (ay * math.cos(q5_4) + ny * math.sin(q5_4) * math.cos(q6_4) - oy * math.sin(q5_4) * math.sin(q6_4))) * math.sin(q1_2)

        t34_4 = -a4 * az * math.cos(q5_4) \
            - a4 * nz * math.sin(q5_4) * math.cos(q6_4) \
            + a4 * oz * math.sin(q5_4) * math.sin(q6_4) \
            - az * d6 - d1 + pz

        cosq3_3 = (t14_3**2 + t34_3**2 - a2**2 - a3**2) / (2 * a2 * a3)
        cosq3_4 = (t14_4**2 + t34_4**2 - a2**2 - a3**2) / (2 * a2 * a3)

        if abs(sinq5_2) <= 1:
            if abs(cosq3_3) <= 1:
                q3_5 = math.acos(cosq3_3) - self.curDH[2].theta
                q3_6 = -math.acos(cosq3_3) - self.curDH[2].theta
            else:
                q3_5 = float('nan'); q3_6 = float('nan')

            if abs(cosq3_4) <= 1:
                q3_7 = math.acos(cosq3_4) - self.curDH[2].theta
                q3_8 = -math.acos(cosq3_4) - self.curDH[2].theta
            else:
                q3_7 = float('nan'); q3_8 = float('nan')
        else:
            q3_5 = q3_6 = q3_7 = q3_8 = float('nan')
        
        g2_1 = a3 * math.sin(q3_1 + self.curDH[2].theta)
        h2_1 = a2 + a3 * math.cos(q3_1 + self.curDH[2].theta)

        g2_2 = a3 * math.sin(q3_2 + self.curDH[2].theta)
        h2_2 = a2 + a3 * math.cos(q3_2 + self.curDH[2].theta)

        g2_3 = a3 * math.sin(q3_3 + self.curDH[2].theta)
        h2_3 = a2 + a3 * math.cos(q3_3 + self.curDH[2].theta)

        g2_4 = a3 * math.sin(q3_4 + self.curDH[2].theta)
        h2_4 = a2 + a3 * math.cos(q3_4 + self.curDH[2].theta)

        if abs(sinq5_1) <= 1:
            q2_1 = math.atan2(h2_1 * t34_1 - g2_1 * t14_1, h2_1 * t14_1 + g2_1 * t34_1) - self.curDH[1].theta
            q2_2 = math.atan2(h2_2 * t34_1 - g2_2 * t14_1, h2_2 * t14_1 + g2_2 * t34_1) - self.curDH[1].theta
            q2_3 = math.atan2(h2_3 * t34_2 - g2_3 * t14_2, h2_3 * t14_2 + g2_3 * t34_2) - self.curDH[1].theta
            q2_4 = math.atan2(h2_4 * t34_2 - g2_4 * t14_2, h2_4 * t14_2 + g2_4 * t34_2) - self.curDH[1].theta
        else:
            q2_1 = float('nan')
            q2_2 = float('nan')
            q2_3 = float('nan')
            q2_4 = float('nan')

        g2_5 = a3 * math.sin(q3_5 + self.curDH[2].theta)
        h2_5 = a2 + a3 * math.cos(q3_5 + self.curDH[2].theta)

        g2_6 = a3 * math.sin(q3_6 + self.curDH[2].theta)
        h2_6 = a2 + a3 * math.cos(q3_6 + self.curDH[2].theta)

        g2_7 = a3 * math.sin(q3_7 + self.curDH[2].theta)
        h2_7 = a2 + a3 * math.cos(q3_7 + self.curDH[2].theta)

        g2_8 = a3 * math.sin(q3_8 + self.curDH[2].theta)
        h2_8 = a2 + a3 * math.cos(q3_8 + self.curDH[2].theta)

        if abs(sinq5_2) <= 1:
            q2_5 = math.atan2(h2_5 * t34_3 - g2_5 * t14_3, h2_5 * t14_3 + g2_5 * t34_3) - self.curDH[1].theta
            q2_6 = math.atan2(h2_6 * t34_3 - g2_6 * t14_3, h2_6 * t14_3 + g2_6 * t34_3) - self.curDH[1].theta
            q2_7 = math.atan2(h2_7 * t34_4 - g2_7 * t14_4, h2_7 * t14_4 + g2_7 * t34_4) - self.curDH[1].theta
            q2_8 = math.atan2(h2_8 * t34_4 - g2_8 * t14_4, h2_8 * t14_4 + g2_8 * t34_4) - self.curDH[1].theta
        else:
            q2_5 = float('nan')
            q2_6 = float('nan')
            q2_7 = float('nan')
            q2_8 = float('nan')

        sin234_1 = -math.cos(q6_1) * (ox * math.cos(q1_1) + oy * math.sin(q1_1)) \
                - math.sin(q6_1) * (nx * math.cos(q1_1) + ny * math.sin(q1_1))

        sin234_2 = -math.cos(q6_2) * (ox * math.cos(q1_1) + oy * math.sin(q1_1)) \
                - math.sin(q6_2) * (nx * math.cos(q1_1) + ny * math.sin(q1_1))

        cos234_1 = oz * math.cos(q6_1) + nz * math.sin(q6_1)
        cos234_2 = oz * math.cos(q6_2) + nz * math.sin(q6_2)

        if abs(sinq5_1) <= 1:
            q4_1 = math.atan2(sin234_1, cos234_1) - q2_1 - q3_1 - self.curDH[1].theta
            q4_2 = math.atan2(sin234_1, cos234_1) - q2_2 - q3_2 - self.curDH[1].theta
            q4_3 = math.atan2(sin234_2, cos234_2) - q2_3 - q3_3 - self.curDH[1].theta
            q4_4 = math.atan2(sin234_2, cos234_2) - q2_4 - q3_4 - self.curDH[1].theta
        else:
            q4_1 = q4_2 = q4_3 = q4_4 = float('nan')

        sin234_3 = -math.cos(q6_3) * (ox * math.cos(q1_2) + oy * math.sin(q1_2)) \
                - math.sin(q6_3) * (nx * math.cos(q1_2) + ny * math.sin(q1_2))

        sin234_4 = -math.cos(q6_4) * (ox * math.cos(q1_2) + oy * math.sin(q1_2)) \
                - math.sin(q6_4) * (nx * math.cos(q1_2) + ny * math.sin(q1_2))

        cos234_3 = oz * math.cos(q6_3) + nz * math.sin(q6_3)
        cos234_4 = oz * math.cos(q6_4) + nz * math.sin(q6_4)

        if abs(sinq5_2) <= 1:
            q4_5 = math.atan2(sin234_3, cos234_3) - q2_5 - q3_5 - self.curDH[1].theta
            q4_6 = math.atan2(sin234_3, cos234_3) - q2_6 - q3_6 - self.curDH[1].theta
            q4_7 = math.atan2(sin234_4, cos234_4) - q2_7 - q3_7 - self.curDH[1].theta
            q4_8 = math.atan2(sin234_4, cos234_4) - q2_8 - q3_8 - self.curDH[1].theta
        else:
            q4_5 = q4_6 = q4_7 = q4_8 = float('nan')
        
        # ---------- 汇总结果 ----------
        result = []

        if not (math.isnan(q1_1) or math.isnan(q2_1) or math.isnan(q3_1) or math.isnan(q4_1) or math.isnan(q5_1) or math.isnan(q6_1)):
            result.append(self.NormalizeJointAngles([q1_1, q2_1, q3_1, q4_1, q5_1, q6_1]))

        if not (math.isnan(q1_1) or math.isnan(q2_2) or math.isnan(q3_2) or math.isnan(q4_2) or math.isnan(q5_1) or math.isnan(q6_1)):
            result.append(self.NormalizeJointAngles([q1_1, q2_2, q3_2, q4_2, q5_1, q6_1]))

        if not (math.isnan(q1_1) or math.isnan(q2_3) or math.isnan(q3_3) or math.isnan(q4_3) or math.isnan(q5_2) or math.isnan(q6_2)):
            result.append(self.NormalizeJointAngles([q1_1, q2_3, q3_3, q4_3, q5_2, q6_2]))

        if not (math.isnan(q1_1) or math.isnan(q2_4) or math.isnan(q3_4) or math.isnan(q4_4) or math.isnan(q5_2) or math.isnan(q6_2)):
            result.append(self.NormalizeJointAngles([q1_1, q2_4, q3_4, q4_4, q5_2, q6_2]))

        if not (math.isnan(q1_2) or math.isnan(q2_5) or math.isnan(q3_5) or math.isnan(q4_5) or math.isnan(q5_3) or math.isnan(q6_3)):
            result.append(self.NormalizeJointAngles([q1_2, q2_5, q3_5, q4_5, q5_3, q6_3]))

        if not (math.isnan(q1_2) or math.isnan(q2_6) or math.isnan(q3_6) or math.isnan(q4_6) or math.isnan(q5_3) or math.isnan(q6_3)):
            result.append(self.NormalizeJointAngles([q1_2, q2_6, q3_6, q4_6, q5_3, q6_3]))

        if not (math.isnan(q1_2) or math.isnan(q2_7) or math.isnan(q3_7) or math.isnan(q4_7) or math.isnan(q5_4) or math.isnan(q6_4)):
            result.append(self.NormalizeJointAngles([q1_2, q2_7, q3_7, q4_7, q5_4, q6_4]))

        if not (math.isnan(q1_2) or math.isnan(q2_8) or math.isnan(q3_8) or math.isnan(q4_8) or math.isnan(q5_4) or math.isnan(q6_4)):
            result.append(self.NormalizeJointAngles([q1_2, q2_8, q3_8, q4_8, q5_4, q6_4]))

        select = None
        min_d = 1000
        min_d_idx = 0

        # 计算六个关节位置差的绝对值，之和
        def get_d(ans, q):
            ret = 0
            for i in range(6):
                ret += abs(ans[i] - q[i])

            return ret

        if len(result) > 0:
            # 便利所有解，找最近的解
            for i in range (len(result)):
                ans = result[i]
                d = get_d(ans, q)
                if d < min_d:
                    min_d = d
                    min_d_idx = i
                    select = ans
        
        return select, min_d_idx
    
class MujocoRobot:
    def __init__(self, joint_step_size = 0.02, gripper_step_size = 0.002, ee_step_size = 0.008, ee_roll_size = 0.015):
        # 加载机器人模型
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = mujoco.MjModel.from_xml_path(f"{base_dir}/urdf/urdf/arm.xml")
        self.data = mujoco.MjData(self.model)
        self.q_des = np.copy(self.data.qpos)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.joint_step_size = joint_step_size
        self.gripper_step_size = gripper_step_size
        self.ee_step_size = ee_step_size
        self.ee_roll_size = ee_roll_size

    def update_param(self, qos, gripper):
        self.data.qpos[7:13] = qos
        self.data.qpos[13] = gripper
        self.data.qpos[14] = gripper
        self.data.qvel[:] = 0

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
