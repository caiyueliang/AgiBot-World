import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from typing import Any, Dict, Union
import numpy as np
import torch
import yaml
from PIL import Image
from scripts.agibot_model import create_model
import rclpy
import threading
import json
from cv_bridge import CvBridge
from genie_sim_ros import SimROSNode
from configs.state_vec import STATE_VEC_IDX_MAPPING
import argparse
np.random.seed(0)
from queue import Queue


def fill_in_state(values):
    # Target indices corresponding to your state space
    # In this example: 6 joints + 1 gripper for each arm
    UNI_STATE_INDICES = (
        [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)]
        + [STATE_VEC_IDX_MAPPING["left_gripper_open"]]
        + [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
        + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
    )
    uni_vec = np.zeros(values.shape[:-1] + (128,))
    uni_vec[..., UNI_STATE_INDICES] = values

    return uni_vec


# Initialize the model
def make_policy(config_path, pretrained_model_name_or_path, ctrl_freq):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    pretrained_vision_encoder_name_or_path = (
        "google/siglip-so400m-patch14-384"
    )
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=pretrained_model_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=ctrl_freq,
    )
    return model


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time


class RDTInfer:
    def __init__(
        self,
        model_name_or_path: Union[str, Path],
        config_path="configs/base.yaml",
        ctrl_freq: int = 30,
        task=None,
    ) -> Path:
        self.task = task
        self.model_name_or_path = model_name_or_path
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.ctrl_freq = ctrl_freq
        self.model = make_policy(config_path, model_name_or_path, ctrl_freq)
        lang_embed_path = f"lang/{task}.pt"
        self.text_embeds = torch.load(lang_embed_path).to(self.device)

        if model_name_or_path.split("/")[-1].startswith("checkpoint-"):
            self.save_name = model_name_or_path.split("/")[-2]
            self.step = model_name_or_path.split("/")[-1].split("-")[-1]
        else:
            self.save_name = model_name_or_path.split("/")[-1]
            self.step = "last"

    def predict_action(self, payload1: Dict[str, Any], payload2: Dict[str, Any]):
        # NOTE here we expect two frames of history
        # Assume current frame t, namely paload1: t-1, paload2:t
        # Parse payload components
        images_arr = [
            np.array(payload1["cam_tensor_head_top"], dtype=np.uint8),
            np.array(payload1["cam_tensor_wrist_right"], dtype=np.uint8),
            np.array(payload1["cam_tensor_wrist_left"], dtype=np.uint8),
            np.array(payload2["cam_tensor_head_top"], dtype=np.uint8),
            np.array(payload2["cam_tensor_wrist_right"], dtype=np.uint8),
            np.array(payload2["cam_tensor_wrist_left"], dtype=np.uint8),
        ]
        images = [
            Image.fromarray(arr) if arr is not None else None for arr in images_arr
        ]

        proprio = torch.from_numpy(payload2["state"]).float().unsqueeze(0)  # (1, 128)
        state_indicator = (
            torch.from_numpy(payload2["state_indicator"]).float().unsqueeze(0)
        )  # (1, 128)
        # Inference
        actions = (
            self.model.step(
                proprio=proprio,
                images=images,
                text_embeds=self.text_embeds,
                state_indicator=state_indicator,
            )
            .squeeze(0)
            .cpu()
        )  # [64, 16]

        return actions

    def infer_one_step(self, payload1, payload2, n_actions: int):

        pred_action = self.predict_action(payload1, payload2)

        # only use the first n_actions
        pred_action = pred_action[:n_actions, :]

        return pred_action

    def infer(self, n_actions=64):

        rclpy.init()
        current_path = os.getcwd()
        sim_ros_node = SimROSNode()
        spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
        spin_thread.start()

        bridge = CvBridge()
        count = 0
        SIM_INIT_TIME = 10

        img_h_1 = None
        img_l_1 = None
        img_r_1 = None
        state_1 = None

        img_h_2 = None
        img_l_2 = None
        img_r_2 = None
        state_2 = None

        action_queue = Queue()

        while rclpy.ok():
            img_h_raw = sim_ros_node.get_img_head()
            img_l_raw = sim_ros_node.get_img_left_wrist()
            img_r_raw = sim_ros_node.get_img_right_wrist()
            act_raw = sim_ros_node.get_joint_state()

            if (
                img_h_raw
                and img_l_raw
                and img_r_raw
                and act_raw
                and img_h_raw.header.stamp
                == img_l_raw.header.stamp
                == img_r_raw.header.stamp
            ):
                sim_time = get_sim_time(sim_ros_node)
                if sim_time > SIM_INIT_TIME:
                    print("cur sim time", sim_time)
                    img_h_1 = img_h_2
                    img_l_1 = img_l_2
                    img_r_1 = img_r_2
                    state_1 = state_2

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

                    state_raw = np.array(act_raw.position)
                    state_indicator = fill_in_state(np.ones_like(state_raw))
                    state = fill_in_state(state_raw)

                    img_h_2 = img_h
                    img_l_2 = img_l
                    img_r_2 = img_r
                    state_2 = state

                    if count == 1:
                        continue

                    payload1 = {
                        "cam_tensor_head_top": img_h_1,
                        "cam_tensor_wrist_right": img_r_1,
                        "cam_tensor_wrist_left": img_l_1,
                        "state": state_1,
                        "state_indicator": state_indicator,
                    }
                    payload2 = {
                        "cam_tensor_head_top": img_h_2,
                        "cam_tensor_wrist_right": img_r_2,
                        "cam_tensor_wrist_left": img_l_2,
                        "state": state_2,
                        "state_indicator": state_indicator,
                    }

                    if not action_queue.empty():
                        action = action_queue.get()

                    else:
                        action_chunk = self.infer_one_step(payload1, payload2, n_actions)
                        joint_l = action_chunk[:, :7]
                        gripper_l_ = action_chunk[:, -2]
                        gripper_l = gripper_l_
                        joint_r = action_chunk[:, 7:-2]
                        gripper_r_ = action_chunk[:, -1]
                        gripper_r = gripper_r_

                        for idx, v in enumerate(gripper_l_):

                            if v < 0.5:
                                gripper_l[idx] = 0
                            else:
                                gripper_l[idx] = 1

                        for idx, v in enumerate(gripper_r_):
                            if v < 0.5:
                                gripper_r[idx] = 0
                            else:
                                gripper_r[idx] = 1

                        gripper_l = gripper_l.unsqueeze(-1)
                        gripper_r = gripper_r.unsqueeze(-1)

                        action_chunk = torch.concat(
                            (joint_l, gripper_l, joint_r, gripper_r), dim=1
                        )
                        for action in action_chunk:
                            action_queue.put(action)

                    sim_ros_node.publish_joint_command(action)
                sim_ros_node.loop_rate.sleep()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--task_name", type=str)
    args = parser.parse_args()
    
    checkpoint_path = (
        "checkpoints/finetuned"
    )
    rdtinfer = RDTInfer(checkpoint_path, task=args.task_name)

    pred_action = rdtinfer.infer(n_actions=64)
