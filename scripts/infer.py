import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np

import draccus
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

import rclpy
import time
from cv_bridge import CvBridge

from genie_sim_ros import SimROSNode

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resize_img(img, width, height):
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


def infer(policy):

    rclpy.init()
    sim_ros_node = SimROSNode()
    
    bridge = CvBridge()

    while rclpy.ok():
        rclpy.spin_once(sim_ros_node)

        # 打开图像文件
        img_raw = sim_ros_node.get_img_head()
        if img_raw:
            
            image = bridge.compressed_imgmsg_to_cv2(img_raw, desired_encoding="rgb8")
            lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
            
            action = policy.step(image, lang)
            
            sim_ros_node.publish_joint_command(action)
        else:
            print("skip")
            
        time.sleep(0.1)


@dataclass
class GenerateConfig:

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/finetuned"
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
                  
    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    seed: int = 7  

    action_decoder_path:str = "checkpoints/finetuned/action_decoder.pt"
    window_size: int = 30
    
    n_layers: int = 2
    hidden_dim: int = 1024
    balancing_factor: float = 0.01                     # larger for smoother

    debug: bool = False


@draccus.wrap()
def get_policy(cfg: GenerateConfig) -> None:

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)

    return policy


if __name__ == "__main__":
    
    policy = get_policy()
    infer(policy)
