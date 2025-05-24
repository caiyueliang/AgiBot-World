import os
import sys
sys.path.append("/home/zy/workspace/main/univla")
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import draccus
import torch
import torch.distributed as dist
import cv2
import torch.distributed as dist
import numpy as np
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append("/home/zy/workspace/main/InternVL")
sys.path.append("/home/zy/workspace/main/univla")
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel


def resize_img(img, width, height):
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


def setup_distributed():
    """Initialize distributed training environment"""
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    # Parse environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the distributed environment
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)  # or 'gloo' for CPU

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return local_rank, world_size


def infer(policy):
    
    ########
    import rclpy
    import threading, time
    
    from cv_bridge import CvBridge

    from univla.genie_sim_ros import SimROSNode

    rclpy.init()
    sim_ros_node = SimROSNode()
    
    bridge = CvBridge()

    # spin_thread = threading.Thread(
    #     target=rclpy.spin, args=(sim_ros_node,)
    # )
    # spin_thread.start()
    # lock = threading.Lock()
    # spin_thread.start()

    # exit()

    # rclpy.spin(sim_ros_node)

    ########


    while rclpy.ok():
        rclpy.spin_once(sim_ros_node)

        # 打开图像文件
        img_raw = sim_ros_node.get_img_head()
        if img_raw:
            
            image = bridge.compressed_imgmsg_to_cv2(img_raw, desired_encoding="rgb8")
            image = resize_img(image, 640, 480)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("/home/zy/workspace/main/univla/1.png", image)
            image_array = np.array(image)
            
            lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
            
            action = policy.step(image_array, lang)
            # import ipdb; ipdb.set_trace()
            print(action)
            
            sim_ros_node.publish_joint_command(action)
        else:
            print("skip")
            
        time.sleep(0.1)


@dataclass
class GenerateConfig:

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/zy/workspace/main/univla/checkpoints/run-best-v2"
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
                  
    center_crop: bool = False                        # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"   # Local directory for eval logs
    seed: int = 7  

    action_decoder_path:str = "/home/zy/workspace/main/univla/checkpoints/run-best-v2/action_decoder-15.pt"
    window_size: int = 30
    
    n_layers: int = 2
    hidden_dim: int = 1024
    balancing_factor: float = 0.01                     # larger for smoother

    debug: bool = False


@draccus.wrap()
def get_policy(cfg: GenerateConfig) -> None:

    setup_distributed()

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)

    return policy


if __name__ == "__main__":
    
    policy = get_policy()
    infer(policy)
