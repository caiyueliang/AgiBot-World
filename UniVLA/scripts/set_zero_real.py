import os
import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from robot_interface import RobotNode
import numpy as np
from dataclasses import dataclass
from typing import Union
import draccus


def get_instruction(task_name):

    if task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    else:
        raise ValueError("task does not exist")

    return lang


def init_node_pos(node, task_name):
    # upper head lower waist
    if task_name == "iros_pack_in_the_supermarket": # 1418
        head_action = [
            0.0,
            0.436332306,
            ]
        waist_action = [
            0.52359929,
            0.27,
            ]
        joint_action = [
            -1.11224556,  0.53719825,  0.45914441, -1.23825192,  0.5959, 1.41219366, -0.08660435, 0,
            1.07460594, -0.61097687, -0.2804215, 1.28363943, -0.72993356, -1.4951334, 0.18722105, 0,
        ]

    elif task_name == "iros_heat_the_food_in_the_microwave":
        head_action = [
            0.0,
            0.43633231,
            ]
        waist_action = [
            0.43633204, 
            0.24
            ]
        joint_action = [
            -1.0742743, 0.61099428, 0.279549, -1.28383136, 0.73043954, 1.49532545, -0.1876224, 0,
            1.07420456, -0.61097687, -0.2795839, 1.28395355, -0.73038721, -1.49534285, 0.18760496, 0,
        ]

    else:
        raise ValueError("task does not exist")

    node.publish_head_command(head_action)
    node.publish_waist_command(waist_action)
    node.publish_joint_command(joint_action)


def set_zero(cfg):
    node = RobotNode()
    init_node_pos(node, cfg.task_name)
    node.robot.shutdown()


@dataclass
class GenerateConfig:
    task_name: str = "iros_pack_in_the_supermarket"


@draccus.wrap()
def get_cfg(cfg: GenerateConfig) -> None:
    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    set_zero(cfg)
