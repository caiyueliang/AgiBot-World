from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node

from sensor_msgs.msg import (
    CompressedImage,
    JointState,
)

import threading


QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class SimROSNode(Node):
    def __init__(self, robot_cfg=None, node_name="univla_node"):
        super().__init__(node_name,)

        self.robot_cfg = robot_cfg
        
        # publish
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/joint_command",
            QOS_PROFILE_LATEST,
        )

        # subscribe
        self.sub_img_head = self.create_subscription(
            CompressedImage,
            "/sim/head_img",
            self.callback_rgb_image_head,
            1,
        )

        # msg
        self.lock = threading.Lock()

        # loop
        self.loop_rate = self.create_rate(30.0)
        
        self.img_head = None

    def callback_rgb_image_head(self, msg):
        print(msg.header)
        with self.lock:
            self.img_head = msg

    
    def get_img_head(self):
        with self.lock:
            return self.img_head
        
    def publish_joint_command(self, action):
        
        cmd_msg = JointState()
        cmd_msg.name = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        ]
        cmd_msg.position = [0.0] * len(cmd_msg.name)
        cmd_msg.position[0] = action[0]
        cmd_msg.position[1] = action[1]
        cmd_msg.position[2] = action[2]
        cmd_msg.position[3] = action[3]
        cmd_msg.position[4] = action[4]
        cmd_msg.position[5] = action[5]
        cmd_msg.position[6] = action[6]
        cmd_msg.position[7] = action[7]
        cmd_msg.position[8] = action[8]
        cmd_msg.position[9] = action[9]
        cmd_msg.position[10] = action[10]
        cmd_msg.position[11] = action[11]
        cmd_msg.position[12] = action[12]
        cmd_msg.position[13] = action[13]
        cmd_msg.position[14] = action[14]
        cmd_msg.position[15] = action[15]

        self.pub_joint_command.publish(cmd_msg)
