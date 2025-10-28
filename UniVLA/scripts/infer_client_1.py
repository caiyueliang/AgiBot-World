# client.py
import requests
import base64
from PIL import Image
import numpy as np

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from PIL import Image
from dataclasses import dataclass
from typing import Union

from datetime import datetime
import cv2
import numpy as np
import draccus
import logging

import rclpy
import threading
from cv_bridge import CvBridge

from genie_sim_ros import SimROSNode

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def image_to_base64(img_array):
    # 确保图像是 uint8 类型
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Input is not a valid numpy array.")
    
    # 如果是灰度图，转成三通道（可选，根据你的模型需求）
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 3:
        pass  # 正常彩色图像
    else:
        raise ValueError("Unsupported image format")

    # 编码为 JPEG 格式（可改为 PNG，但 JPEG 更小）
    success, encoded_image = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Could not encode image to JPEG")

    # 转为 base64
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')

def post_request(head_img, wrist_l_img, wrist_r_img, instruction, state):
    # 构造请求
    base_url = cfg.url

    if isinstance(state, np.ndarray):
        state = state.tolist()

    request_data = {
        "head_img": image_to_base64(head_img),
        "wrist_left_img": image_to_base64(wrist_l_img),
        "wrist_right_img": image_to_base64(wrist_r_img),
        "instruction": instruction,
        "state":  state  # 示例关节状态，16维
    }

    # 发送请求
    response = requests.post(base_url, json=request_data)

    print("state:", request_data["state"])
    if response.status_code == 200:
        result = response.json()
        print("Action:", result["action"])
        print("Timestamp:", result["timestamp"])
        return result["action"]
    else:
        print("Error:", response.json())
        return []

def resize_img(img, width, height):
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


def get_instruction(task_name):

    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time


def infer(cfg):
    print("[start infer] ...")
    print(f"[start infer] cfg: {cfg}")
    rclpy.init()
    current_path = os.getcwd()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()

    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 10

    lang = get_instruction(cfg.task_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"frames/{cfg.task_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    save_steps = 50

    print("[start infer] run ...")
    while rclpy.ok():
        # 从仿真节点获取传感器数据
        img_h_raw = sim_ros_node.get_img_head()           # 头部摄像头图像（原始 ROS 消息）
        img_l_raw = sim_ros_node.get_img_left_wrist()     # 左手腕摄像头图像
        img_r_raw = sim_ros_node.get_img_right_wrist()    # 右手腕摄像头图像
        act_raw = sim_ros_node.get_joint_state()          # 当前关节状态（位置、速度等）

        # 检查所有传感器数据是否有效，且时间戳一致（保证同步）
        if (
            img_h_raw
            and img_l_raw
            and img_r_raw
            and act_raw
            and img_h_raw.header.stamp
            == img_l_raw.header.stamp
            == img_r_raw.header.stamp
        ):
            sim_time = get_sim_time(sim_ros_node)                           # 获取当前仿真时间
            if sim_time > SIM_INIT_TIME:
                # print("cur sim time", sim_time)
                count = count + 1
                img_h = bridge.compressed_imgmsg_to_cv2(
                    img_h_raw, desired_encoding="rgb8"
                )
                img_l = bridge.compressed_imgmsg_to_cv2(
                    img_l_raw, desired_encoding="rgb8"
                )
                img_r = bridge.compressed_imgmsg_to_cv2(
                    img_r_raw, desired_encoding="rgb8"
                )

                if count % save_steps == 0:
                    logging.warning(f"[head_rgb] {type(img_h)}, {img_h.shape}")
                    logging.warning(f"[wrist_l_rgb] {type(img_l)}, {img_l.shape}")
                    logging.warning(f"[wrist_r_rgb] {type(img_r)}, {img_r.shape}")
                    logging.warning(f"[lang] {lang}")
                    logging.warning(f"[state] {state}")

                    img_h_pil = Image.fromarray(img_h)
                    img_h_pil.save(f'{save_dir}/head_{count:05d}.png')
                    img_l_pil = Image.fromarray(img_l)
                    img_l_pil.save(f'{save_dir}/wrist_l_{count:05d}.png')
                    img_r_pil = Image.fromarray(img_r)
                    img_r_pil.save(f'{save_dir}/wrist_r_{count:05d}.png')
                    print(f"Saved frame at count = {count}")

                state = np.array(act_raw.position)                          # 提取当前机器人的关节位置作为本体感觉（proprioception）
                # if cfg.with_proprio:
                #     action = policy.step(img_h, img_l, img_r, lang, state)  # 传入图像、语言、状态
                # else:
                #     action = policy.step(img_h, img_l, img_r, lang)         # 仅传入图像和语言
                action = post_request(head_img=img_h, wrist_l_img=img_l, wrist_r_img=img_r, instruction=lang, state=state)

                print(f"[sim_time] {sim_time};\n [state] {state}; \n[action] {action}")

                sim_ros_node.publish_joint_command(action)                  # 将模型输出的动作发送给机器人执行
                sim_ros_node.loop_rate.sleep()                              # 按照设定的循环频率休眠，保持稳定控制周期


@dataclass
class GenerateConfig:
    url: str = "http://localhost:8888/infer"

    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/finetuned"

    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False  # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"  # Local directory for eval logs
    seed: int = 7

    action_decoder_path: str = "checkpoints/finetuned/action_decoder.pt"
    window_size: int = 30

    n_layers: int = 2
    hidden_dim: int = 1024

    with_proprio: bool = True

    smooth: bool = False
    balancing_factor: float = 0.1  # larger for smoother

    task_name: str = "iros_stamp_the_seal"


if __name__ == "__main__":
    cfg = draccus.parse(GenerateConfig)
    infer(cfg)
