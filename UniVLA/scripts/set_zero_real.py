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


def init_node_pos(node, task_name):
    # upper head lower waist
    if task_name == "Pack groceries": # 1418
        head_action = [
            0.0,
            0.436332306,
            ]
        waist_action = [
            0.52359929,
            27,
            ]
        joint_action = [
            -1.11224556,  0.53719825,  0.45914441, -1.23825192,  0.5959, 1.41219366, -0.08660435, 0,
            1.07460594, -0.61097687, -0.2804215, 1.28363943, -0.72993356, -1.4951334, 0.18722105, 0,
        ]

    elif task_name == "Microwave the food": # 881
        head_action = [
            0.0,
            0.43633231,
            ]
        waist_action = [
            0.43633204, 
            24
            ]
        joint_action = [
            -1.0742743, 0.61099428, 0.279549, -1.28383136, 0.73043954, 1.49532545, -0.1876224, 0,
            1.07420456, -0.61097687, -0.2795839, 1.28395355, -0.73038721, -1.49534285, 0.18760496, 0,
        ]

    elif task_name == "Pack items from conveyor": # 858
        head_action = [
            0.0,
            0.4363,
            ]
        waist_action = [
            0.5236, 
            24,
            ]
        joint_action = [
            -1.085, 0.5951, 0.3214, -1.279, 0.7025, 1.479, -0.1656, 0,
            1.075, -0.6117, -0.2797, 1.282, -0.7310, -1.495, 0.1868, 0,
        ]

    elif task_name == "Fold short sleeves": # 2195
        head_action = [
            0.0,
            0.4363323055555555,
            ]
        waist_action = [
            0.8901176920412174, 
            45.98676300048828
            ]
        joint_action = [
            -1.0742219686508179, 0.6111513376235962, 0.27946174144744873, -1.2839535474777222, 0.7303872108459473, 1.495360255241394, -0.18760496377944946, 0, 1.0742743015289307, -0.6110466122627258, -0.2794792056083679, 1.2838836908340454, -0.7303697466850281, -1.4952380657196045, 0.18762239813804626, 0,
        ]

    elif task_name == "Restock the hanging area": # 3173
        head_action = [
            0.0,
            0.4363,
            ]
        waist_action = [
            0.1920, 
            31
            ]
        joint_action = [
            -1.686, 0.9457, 1.330, -0.8735, 0.1478, 1.252, 0.03354, 0,
            1.815, -0.6138, -1.393, 0.8388, -0.1517, -1.466, -0.06992, 0,
        ]

    elif task_name == "Pour water": # 3377
        head_action = [
            0.0,
            0.43633231,
            ]
        waist_action = [
            0.52359771, 
            29.999931
            ]
        joint_action = [
            -1.03729784, 0.58743685, 0.27705365, -1.23694324, 0.70110613, 1.44067192, -0.17989205, 0,
            1.07420456, -0.611099, -0.27960137, 1.28388369, -0.73043954, -1.49543011, 0.1876224, 0,
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
