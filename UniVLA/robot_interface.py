import ruckig
import time
import numpy as np
from a2d_sdk.robot import RobotDds, CosineCamera


class RobotNode:
    """
    单线程 / 无锁版本 SimNode，接口与 ROS 版本保持一致。
    每次 get_* 都实时拉取最新数据；publish 直接下发。
    """

    def __init__(self, hz: float = 30.0):
        self.robot   = RobotDds()
        self.camera  = CosineCamera(["head", "hand_left", "hand_right"])
        self.period  = 1.0 / hz
        self._t0     = time.time()

        # 先等底层数据就绪
        self._wait_until_ready()

    # -------------------------------------
    def _wait_until_ready(self):
        """阻塞直到拿到第一张图和关节"""
        while True:
            img, _ = self.camera.get_latest_image("head")
            pos, _ = self.robot.arm_joint_states()
            grip, _ = self.robot.gripper_states()
            if img is not None and pos is not None and grip is not None:
                break
            time.sleep(0.01)

    # ------------ 原接口 ------------------
    def get_img_head(self):
        img, _ = self.camera.get_latest_image("head")
        return img[:, :, ::-1] if img is not None else None

    def get_img_left_wrist(self):
        img, _ = self.camera.get_latest_image("hand_left")
        return img[:, :, ::-1] if img is not None else None

    def get_img_right_wrist(self):
        img, _ = self.camera.get_latest_image("hand_right")
        return img[:, :, ::-1] if img is not None else None

    def get_joint_state(self):
        """返回 16 维 [左7 左夹 右7 右夹]"""
        pos, _  = self.robot.arm_joint_states()
        grip, _ = self.robot.gripper_states()
        if pos is None or grip is None:
            return None
        left_g  = max(0.0, min(1.0, 0.8 - grip[0])) if grip[0] is not None else 0.0
        right_g = max(0.0, min(1.0, 0.8 - grip[1])) if grip[1] is not None else 0.0
        return pos[:7] + [left_g] + pos[7:] + [right_g]

    def publish_joint_command(self, action):
        """action: 16 维"""
        target_positions = np.concatenate((action[:7], action[8:15]))

        # Get current joint states
        current_positions, _ = self.robot.arm_joint_states()
        if not current_positions:
            raise Exception("Failed to get arm joint states")

        print(f"Planning trajectory from current positions to target...")
        print(f"Current positions: {current_positions}")
        print(f"Target positions: {target_positions}")

        dof = 14  # Degrees of freedom
        interval = 0.01  # Time interval
        rk = ruckig.Ruckig(dof, interval)
        rk_input = ruckig.InputParameter(dof)
        rk_output = ruckig.OutputParameter(dof)

        # Set current state
        rk_input.current_position = current_positions
        rk_input.current_velocity = [0.0] * dof
        rk_input.current_acceleration = [0.0] * dof

        # Set target state
        rk_input.target_position = target_positions
        rk_input.target_velocity = [0.0] * dof
        rk_input.target_acceleration = [0.0] * dof

        # Set motion constraints
        rk_input.max_velocity = [2.0] * dof
        rk_input.max_acceleration = [1.0] * dof
        rk_input.max_jerk = [5.0] * dof

        # Generate trajectory
        print("Generating trajectory...")
        trajs = []
        while rk.update(rk_input, rk_output) == ruckig.Result.Working:
            trajs.append(rk_output.new_position)
            rk_output.pass_to_input(rk_input)
        print(f"Generated {len(trajs)} trajectory points")
        print("Executing trajectory...")
        for i, traj in enumerate(trajs):
            print(f"Executing point {i+1}/{len(trajs)}")
            self.robot.move_arm(traj)
            time.sleep(interval)
        time.sleep(0.5)

        self.robot.move_gripper([action[7], action[15]])

        print("Trajectory test completed successfully!")

    def publish_head_command(self, target_positions):
        print("开始头部运动测试...")
        # 获取当前头部关节状态
        current_positions, _ = self.robot.head_joint_states()
        if not current_positions:
            raise Exception("获取头部关节状态失败")
        print(f"当前头部位置: {current_positions}")
        print(f"目标头部位置: {target_positions}")
        self.robot.move_head(target_positions)
        time.sleep(2)
        print("头部运动完成。\n")

    def publish_waist_command(self, target_positions):
        print("开始腰部运动测试...")
        current_positions, _ = self.robot.waist_joint_states()
        if not current_positions:
            raise Exception("获取腰部关节状态失败")
        print(f"当前腰部位置: {current_positions}")
        print(f"目标腰部位置: {target_positions}")
        self.robot.move_waist(target_positions)
        time.sleep(2)
        print("腰部运动完成。\n")

    def get_clock(self):
        return time.time() - self._t0

    def loop_rate(self):
        time.sleep(self.period)

    def shutdown(self):
        self.robot.shutdown()
