import time
import math
from .DM_CAN import *
import serial
from .RobotKinematics import RobotKinematics

class Matrix3x3:
    def __init__(self, mat=None):
        if mat is None:
            self.M = np.zeros((3, 3), dtype=float)
        else:
            # Extract top-left 3x3 from a 4x4 matrix (assuming mat is np.array)
            self.M = mat[:3, :3]

    def transpose(self):
        t = Matrix3x3()
        t.M = self.M.T
        return t

    def multiply(self, v):
        return np.dot(self.M, v)

    def multiply_vector(self, I, v):
        return np.dot(I, v)


class InverseDynamics:
    @staticmethod
    def dh_trans(alpha, a, d, theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)
        return np.array([
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

    @staticmethod
    def cross(a, b):
        return np.cross(a, b)

    @staticmethod
    def mat3x3_mul_vec3(mat, vec):
        return np.dot(mat, vec)

    @staticmethod
    def mat3x3_mul(A, B):
        return np.dot(A, B)

    @staticmethod
    def inv_dyn2(theta, theta_d, theta_dd, f_external=None):
        n = 6
        # DH parameters formatted as a table-like structure
        # Joint | alpha (rad) | a (m)    | d (m) | theta (rad)
        # 1     | 0           | 0        | 0.1281| theta[0]
        # 2     | pi/2        | 0        | 0     | theta[1] + pi
        # 3     | 0           | 0.37343  | 0     | theta[2] - math.radians(161.26)
        # 4     | 0           | 0.3049   | 0     | theta[3] - math.radians(18.74)
        # 5     | -pi/2       | 0.07143  | 0     | theta[4] - pi/2
        # 6     | -pi/2       | 0        | 0.026 | theta[5]
        dh_table = [
            [0,           0,         0.1205, theta[0]],
            [math.pi/2,   0,         0,      theta[1] + math.pi],
            [0,           0.27722,   0,      theta[2] - math.radians(166.26)],
            [0,           0.25216,    0,      theta[3] - math.radians(13.74)],
            [-math.pi/2,  0.0743,   0,      theta[4] - math.pi / 2],
            [-math.pi/2,  0,         0.03,  theta[5]]
        ]

        alp = np.array([row[0] for row in dh_table])
        a = np.array([row[1] for row in dh_table])
        d = np.array([row[2] for row in dh_table])
        th = np.array([row[3] for row in dh_table])

        w0 = np.array([0.0, 0.0, 0.0])
        wd0 = np.array([0.0, 0.0, 0.0])
        vd0 = np.array([0.0, 0.0, 9.8])
        z = np.array([0.0, 0.0, 1.0])

        # Massess
        m = np.array([0.461, 0.770, 0.689, 0.390, 0.327, 0.51]) #最后一个不带夹爪是0.001，带夹爪是0.5

        # Inertia matrices
        I = [
            np.array([
                [0.0001, -2.9664e-10, 6.4475e-07],
                [-2.9664e-10, 0.001, 4.9794e-09],
                [6.4475e-07, 4.9794e-09, 0.001]
            ], dtype=float),
            np.array([
                [0.001, -0.001, 0.00],
                [-0.001, 0.089, 1.0034e-07],
                [0.00, 1.0034e-07, 0.088]
            ], dtype=float),
            np.array([
                [0.001, 0.002, -0.000],
                [0.002, 0.046, 0.0],
                [-0.000, 0.0, 0.047]
            ], dtype=float),
            np.array([
                [0.002, 0.002, 0.0],
                [0.002, 0.002, 0.0],
                [0.0, 0.0, 0.004]
            ], dtype=float),
            np.array([
                [0.000, -1.6995e-21, 5.2609e-22],
                [-1.6995e-21, 5.8355e-06, 4.828e-08],
                [5.2609e-22, 4.828e-08, 4.035e-06]
            ], dtype=float),
            np.array([
                [0.002, 2.4358e-20, -9.8167e-21],
                [2.4358e-20, 0.002, -8.3423e-21],
                [-9.8167e-21, -8.3423e-21, 0.0001]
            ], dtype=float)
        ]

         # Centers of mass
        pc = [
            np.array([0.000, 0.002, -0.005]),
            np.array([0.204, -0.00, -0.001]),
            np.array([0.180, 0.012,  0.000]),
            np.array([0.061, 0.0560, 0.000]),
            np.array([0.000, 0.005, 0.001]),
            np.array([0.000, 0.00, 0.024])    #最后一个不带夹爪是[0.000, 0.00, 0.001]，带夹爪是[0.000, 0.00, 0.034]
        ]

        # T, R, Rt, p
        T = [None] * n
        R = [None] * n
        Rt = [None] * n
        p = [None] * (n + 1)  # p[6] = zero
        for i in range(n):
            T[i] = InverseDynamics.dh_trans(alp[i], a[i], d[i], th[i])
            R[i] = Matrix3x3(T[i])
            Rt[i] = R[i].transpose()
            p[i] = T[i][:3, 3]
        p[n] = np.zeros(3)

        # Forward recursion
        w = [None] * (n + 1)
        w[0] = w0
        wd = [None] * (n + 1)
        wd[0] = wd0
        vd = [None] * (n + 1)
        vd[0] = vd0
        F = [None] * n
        N = [None] * n
        for i in range(n):
            w[i + 1] = Rt[i].multiply(w[i]) + theta_d[i] * z
            wd[i + 1] = Rt[i].multiply(wd[i]) + np.cross(Rt[i].multiply(w[i]), theta_d[i] * z) + theta_dd[i] * z
            vd[i + 1] = Rt[i].multiply(np.cross(wd[i], p[i]) + np.cross(w[i], np.cross(w[i], p[i])) + vd[i])
            vc = np.cross(wd[i + 1], pc[i]) + np.cross(w[i + 1], np.cross(w[i + 1], pc[i])) + vd[i + 1]
            F[i] = m[i] * vc
            N[i] = np.dot(I[i], wd[i + 1]) + np.cross(w[i + 1], np.dot(I[i], w[i + 1]))

        # Backward recursion
        if f_external is None:
            f_next = np.zeros(3)
            n_next = np.zeros(3)
        else:
            f_next = -np.array([f_external[0], f_external[1], f_external[2]])
            n_next = -np.array([f_external[3], f_external[4], f_external[5]])

        tau = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if i < 5:
                fi = R[i + 1].multiply(f_next) + F[i]
                ni = N[i] + R[i + 1].multiply(n_next) + np.cross(pc[i], F[i]) + np.cross(p[i + 1], R[i + 1].multiply(f_next))
            else:
                fi = f_next + F[i]
                ni = N[i] + n_next + np.cross(pc[i], F[i]) + np.cross(p[i + 1], f_next)
            tau[i] = np.dot(ni, z)
            f_next = fi
            n_next = ni

        return tau

class RobotController:
    def __init__(self, port='COM13', baudrate=921600, type="Master"):
        self.serial_device = serial.Serial(port, baudrate, timeout=0.5)

        self.Motor1 = Motor(DM_Motor_Type.DM4340, 0x01, 0x11)
        self.Motor2 = Motor(DM_Motor_Type.DM4340, 0x02, 0x12)
        self.Motor3 = Motor(DM_Motor_Type.DM4340, 0x03, 0x13)
        self.Motor4 = Motor(DM_Motor_Type.DM4310, 0x04, 0x14)
        self.Motor5 = Motor(DM_Motor_Type.DM4310, 0x05, 0x15)
        self.Motor6 = Motor(DM_Motor_Type.DM4310, 0x06, 0x16)
        self.joints = [self.Motor1, self.Motor2, self.Motor3, self.Motor4, self.Motor5, self.Motor6]

        self.Motor7 = Motor(DM_Motor_Type.DM4310, 0x07, 0x17)
        self.gripper = self.Motor7

        self.RobotCtrl = MotorControl(self.serial_device)
        for joint in self.joints:
            self.RobotCtrl.addMotor(joint)
        self.RobotCtrl.addMotor(self.gripper)


        # 反转轴：2,3,4,5（基于0的索引：1,2,3,4）
        self.inverted_axes = [1, 2, 3, 4]  # 需要反转的电机索引，从0开始

        self.type = type

        # 关节范围
        # self.arm = RobotKinematics()
        # self.joint_ranges = self.arm.joint_ranges
    def close_serial(self):
        """关闭串口设备，释放资源"""
        if hasattr(self, 'serial_device') and self.serial_device.is_open:
            self.serial_device.close()
            print(f"Serial port {self.serial_device.port} closed successfully")
        else:
            print("Serial port is not open or does not exist")


    def enable(self):
        for joint in self.joints:
            self.RobotCtrl.enable(joint)
            time.sleep(0.1)
            self.RobotCtrl.refresh_motor_status(joint)
            # print("joint state:", joint.getState())
            if joint.getState() == 0:
                return False
        
        self.RobotCtrl.enable(self.gripper)
        time.sleep(0.1)
        self.RobotCtrl.refresh_motor_status(self.gripper)
        # print("gripper state:", self.gripper.getState())
        if self.gripper.getState() == 0:
            return False
    
        print(f"{self.type} Arm is enabled")
        return True

    def disable(self):
        for joint in self.joints:
            self.RobotCtrl.disable(joint)
            time.sleep(0.1)
            # print("joint state:", joint.getState())
            self.RobotCtrl.refresh_motor_status(joint)
            if joint.getState() != 0:
                return False
        
        self.RobotCtrl.disable(self.gripper)
        time.sleep(0.1)
        self.RobotCtrl.refresh_motor_status(self.gripper)
        # print("gripper state:", self.gripper.getState())
        if self.gripper.getState() != 0:
            return False
        
        print(f"{self.type} Arm is disabled")
        return True

    def get_motor_params(self, motor):
        self.RobotCtrl.enable(motor)
        print("slave ID:",self.RobotCtrl.read_motor_param(motor,DM_variable.ESC_ID))
        print("Master ID:",self.RobotCtrl.read_motor_param(motor,DM_variable.MST_ID))
        print("Work mode:",self.RobotCtrl.read_motor_param(motor,DM_variable.CTRL_MODE))
        print("ACC:",self.RobotCtrl.read_motor_param(motor,DM_variable.ACC))
        print("DEC:",self.RobotCtrl.read_motor_param(motor,DM_variable.DEC))
        print("V Kp: {:.5g}".format(self.RobotCtrl.read_motor_param(motor, DM_variable.KP_ASR)))
        print("V Ki: {:.5g}".format(self.RobotCtrl.read_motor_param(motor, DM_variable.KI_ASR)))
        print("Pos Kp: {:.5g}".format(self.RobotCtrl.read_motor_param(motor, DM_variable.KP_APR)))
        print("Pos Ki: {:.5g}".format(self.RobotCtrl.read_motor_param(motor, DM_variable.KI_APR)))
        print("Cur q: {:.5g}".format(motor.getPosition()))
        self.RobotCtrl.disable(motor)

    # def set_motor_param(self, motor, acc, dec, kp_asr, ki_asr, kp_apr, ki_apr):
    #     params = [acc, dec, kp_asr, ki_asr, kp_apr, ki_apr]
    #     params_name = [DM_variable.ACC, DM_variable.DEC, DM_variable.KP_ASR, DM_variable.KI_ASR, DM_variable.KP_APR, DM_variable.KI_APR]
    #     for i in range(len(params)):
    #         self.RobotCtrl.change_motor_param(motor, params_name[i], params[i])
    #         self.RobotCtrl.refresh_motor_status(motor)
    #         cur_param = self.RobotCtrl.read_motor_param(motor, params_name[i])
    #         if abs(cur_param - params[i]) > 1e-8:
    #             print(cur_param)
    #             return False

    #     return True

    def set_motor_mode(self, motor, mode):
        """
        设置单个电机控制模式。
        mode: Control_Type 枚举，或字符串 'mit' / 'pos_vel'
        """
        if isinstance(mode, str):
            mode_map = {'mit': Control_Type.MIT, 'pos_vel': Control_Type.POS_VEL}
            mode = mode_map.get(mode, mode)
        curr_mode = self.RobotCtrl.read_motor_param(motor, DM_variable.CTRL_MODE)
        if curr_mode != mode:
            self.RobotCtrl.change_motor_param(motor, DM_variable.CTRL_MODE, mode)
            self.RobotCtrl.refresh_motor_status(motor)
        curr_mode = self.RobotCtrl.read_motor_param(motor, DM_variable.CTRL_MODE)
        if curr_mode != mode:
            return False
        return True
    

    # def set_mode(self):
    #     for i, joint in enumerate(self.joints):
    #         if not self.set_motor_mode(joint, Control_Type.POS_VEL):
    #             print(f"joint {i} set failed")
    #             return False
        
    #     if not self.set_motor_mode(self.gripper, Control_Type.Torque_Pos):
    #         return False
        
    #     return True
    
    def set_mit_mode(self):
        for i, joint in enumerate(self.joints):
            if not self.set_motor_mode(joint, Control_Type.MIT):
                print(f"joint {i} set failed")
                return False
        
        if not self.set_motor_mode(self.gripper, Control_Type.MIT):
            return False
        
        return True
    

    def set_pos_vel_mode(self):
        for i, joint in enumerate(self.joints):
            if not self.set_motor_mode(joint, Control_Type.POS_VEL):
                print(f"joint {i} set failed")
                return False
        
        if not self.set_motor_mode(self.gripper, Control_Type.Torque_Pos):
            return False
        
        return True



    # def set_joint_param(self):
    #     for joint in self.joints:
    #         if joint.MotorType == DM_Motor_Type.DM4340:
    #             if not self.set_motor_param(motor=joint, acc=8, dec=-8, kp_asr=0.000884, ki_asr=0.005, kp_apr=500, ki_apr=0.0001):
    #                 return False
    #         elif joint.MotorType == DM_Motor_Type.DM4310:
    #             if not self.set_motor_param(motor=joint, acc=10, dec=-10, kp_asr=0.00172, ki_asr=0.002, kp_apr=100, ki_apr=0.01):
    #                 return False
    #         elif joint.MotorType == DM_Motor_Type.DM6248:
    #             if not self.set_motor_param(motor=joint, acc=5, dec=-5, kp_asr=0.00068, ki_asr=0.001, kp_apr=400, ki_apr=0.001):
    #                 return False

    #     return True


    def set_zero(self) -> bool:
        ret = True
        for joint in self.joints:
            self.RobotCtrl.enable(joint)
            self.RobotCtrl.set_zero_position(joint)
            self.RobotCtrl.refresh_motor_status(joint)
            q = joint.getPosition()
            if abs(q) > 1e-2:
                ret = False
            self.RobotCtrl.disable(joint)

        self.RobotCtrl.enable(self.gripper)
        self.RobotCtrl.set_zero_position(self.gripper)
        self.RobotCtrl.refresh_motor_status(self.gripper)
        q = self.gripper.getPosition()
        if abs(q) > 1e-2:
            ret = False
        self.RobotCtrl.disable(self.gripper)
        return ret

    def _apply_inversion(self, joint_angles):
        adjusted = joint_angles.copy()
        for idx in self.inverted_axes:
            if idx < len(adjusted):
                adjusted[idx] = -adjusted[idx]
        return adjusted

    def _invert_read(self, joint_angles):
        adjusted = joint_angles.copy()
        for idx in self.inverted_axes:
            if idx < len(adjusted):
                adjusted[idx] = -adjusted[idx]
        return adjusted

    def get_current_joint_angles(self):
        positions = []
        for joint in self.joints:
            self.RobotCtrl.refresh_motor_status(joint)
            positions.append(round(float(joint.getPosition()), 7))

        return self._invert_read(positions)
    
    def get_current_gripper_angles(self):
        self.RobotCtrl.refresh_motor_status(self.gripper)
        gripper = round(float(self.gripper.getPosition()), 7)
        return gripper
    
    def set_joint_angles(self, q, v):
        """
        设置关节角度，可传入统一速度（标量）或每个关节独立速度（列表）。
        v: float 或 list[float]，当为标量时所有关节使用相同速度，
           当为列表时长度为6，分别对应6个关节的速度。
        """
        q = self._apply_inversion(q)
        # 统一速度：转换为列表
        if isinstance(v, (int, float)):
            v_list = [float(v)] * 6
        else:
            v_list = list(v)
        for i, joint in enumerate(self.joints):
            self.RobotCtrl.control_Pos_Vel(joint, q[i], v_list[i])
    
    def set_gripper_angles(self, gripper_angle, v, tau_limit = 0.1):
        if self.type == 'follower':
            self.RobotCtrl.control_pos_force(self.gripper, gripper_angle, v, tau_limit)
        else:
            print("Only Slave Arm can set gripper")

    # def check_joint_limits(self, joint_angles):
    #     """检查关节角度是否在限程内。"""
    #     for i, angle in enumerate(joint_angles):
    #         min_val, max_val = self.joint_ranges[i]  # 获取最小和最大值
    #         angle_rounded = round(angle, 2)  # 保留两位小数判断，避免精度误触
    #         if not (min_val <= angle_rounded <= max_val):
    #             raise ValueError(
    #                 f"关节 {i + 1} 超出范围: 原始值 {angle:.3f}（保留两位 {angle_rounded}）不在 [{min_val}, {max_val}] 内"
    #             )
    #     return True

    def gravity_compensation(self):
        """
        重力补偿 + 关节4阻尼控制。
        damping_coeff: 阻尼系数，作用于关节4（J4），力矩 = -速度 × damping_coeff
        """
        if self.type == 'Grivity_arm' or self.type == 'leader':
            q = []
            dq = []
            for joint in self.joints:
                q.append(joint.getPosition())
                dq.append(joint.getVelocity())
            theta = np.array(self._invert_read(q))
            theta_d = np.array(self._invert_read(dq))  # 速度也要反转

            theta_dd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            f_external = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            tau = InverseDynamics.inv_dyn2(theta, theta_d, theta_dd, f_external)
                  # 将数组中的每个元素格式化为3位小数
            formatted_tau = [f"{x:.3f}" for x in tau]
            #print(f"Gravity compensation torques: {formatted_tau}")
            # 关节4（索引3）添加阻尼项：tau = -v × damping_coeff
            #tau[3] -= theta_d[3] * 0  # 阻尼系数0.1，可以根据需要调整
            tau[1] = tau[1] * 0.7 # joint 3 compensation scale
            tau[2] = tau[2] * 0.5
            tau[3] = tau[3] * 0.7

            for i, joint in enumerate(self.joints):
                if i in self.inverted_axes:
                    self.RobotCtrl.controlMIT(joint, 0, 0, 0, 0, -tau[i])
                else:
                    self.RobotCtrl.controlMIT(joint, 0, 0, 0, 0, tau[i])

        else:
            print("Only leader Arm or Grivity_arm supports Gravity Compensation!!!")