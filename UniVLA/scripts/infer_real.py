import os
import sys
from pathlib import Path
import time
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

from robot_interface import RobotNode
import numpy as np
from dataclasses import dataclass
from typing import Union
import draccus


#####################真机比赛任务的允许评测时间######################
# Pack groceries：90秒
# Microwave the food：150秒
# Pack items from conveyor：90秒
# Fold short sleeves：150秒
# Restock the hanging area：60秒
# Pour water：60秒
#################################################################

#####################真机比赛任务的桌面设置######################
# Pack groceries：桌面高度74cm, 距离机器人50cm
# Microwave the food：桌面高度65-75cm（训练数据带桌面高度泛化）, 距离机器人53cm
# Pack items from conveyor：桌面高度55cm（料框放在桌面上）, 距离机器人33cm
# Fold short sleeves：桌面高度75cm, 距离机器人43cm
# Restock the hanging area：货架高度, 距离机器人82cm
# Pour water：桌面高度75cm, 距离机器人51cm
#################################################################


def get_instruction(task_name):

    if task_name == "Pack groceries": # 1418
        lang = "\
            Pick up the green small bag potato chips on the table with the right arm.;\
            Place the held green small bag potato chips into the felt bag on the table with the right arm.;\
            Pick up the yellow banana candy on the table with the right arm.;Place the held yellow banana candy into the felt bag on the table with the right arm.;\
            Place the held yellow banana candy into the felt bag on the table with the right arm.;\
            Pick up the grape juice on the table with the right arm.;\
            Place the held grape juice into the felt bag on the table with the right arm."

    elif task_name == "Microwave the food": # 881
        lang = "\
            Open the door of the microwave on the table with both arms.;\
            Pick up the plate containing pasta on the table with the right arm.;\
            Place the tray containing pasta into the microwave with the right arm.;\
            Push the misplaced plate into the microwave on the table with the right arm.;\
            Close the microwave door on the table with the left arm.;\
            Press the start button on the right side of the microwave on the table with the right arm."

    elif task_name == "Pack items from conveyor": # 858
        lang = "\
            Pick up the white small bottle shower gel on the conveyor belt with the right arm.;\
            Place the held white small bottle shower gel into the box with the right arm.;\
            Pick up the blue facial cleanser on the conveyor belt with the right arm.;\
            Place the held blue facial cleanser into the box with the right arm.;\
            Pick up the white bottled care solution on the conveyor belt with the right arm.;\
            Place the held white bottled care solution into the box with the right arm."

    elif task_name == "Fold short sleeves": # 2195
        lang = "\
            Fold the lower hem and collar of the clothes on the table with both arms.;\
            Pull the clothes on the table to the edge with both arms.;\
            Fold the collar of the clothes on the table with both arms.;\
            Fold the clothes on the table with the right arm."

    elif task_name == "Restock the hanging area": # 3173
        lang = "\
            Complete the restocking task by picking up the green packaged jelly from the restocking box with the left arm.;\
            Use the left arm to neatly hang the green bagged jelly being held onto the corresponding product area of the shelf.;\
            Use the right arm to neatly hang the red packaged chicken feet onto the corresponding section of the shelf."

    elif task_name == "Pour water": # 3377
        lang = "\
            Pick up the kettle on the table with the right arm.;\
            Pour water into the cup on the table with the held kettle using the right arm.;\
            Place the held kettle on the table with the right arm."

    else:
        raise ValueError("task does not exist")

    return lang


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


def infer(policy, cfg):
    node = RobotNode()

    INIT_TIME = 2.0
    lang = get_instruction(cfg.task_name)

    init_node_pos(node, cfg.task_name)
    time.sleep(5)

    count = 0
    while True:
        # import ipdb;ipdb.set_trace()
        img_head = node.get_img_head()
        img_left = node.get_img_left_wrist()
        img_right = node.get_img_right_wrist()
        joint_state = node.get_joint_state()

        # 等待所有数据就绪
        if img_head is None or img_left is None or img_right is None or joint_state is None:
            time.sleep(0.01)
            continue

        cur_time = node.get_clock()  # 用时间戳或简单计时
        if cur_time < INIT_TIME:
            time.sleep(0.01)
            continue

        count += 1
        state = np.array(joint_state)

        if cfg.with_proprio:
            action_queue = policy.step(img_head, img_left, img_right, lang, state)
        else:
            action_queue = policy.step(img_head, img_left, img_right, lang)

        for action in action_queue:
            # action = gaussian_filter1d(action, sigma=3)
            node.publish_joint_command(action)
            time.sleep(0.05)

        node.loop_rate()      # 30 Hz

    node.robot.shutdown()


@dataclass
class GenerateConfig:

    name = "run-onsite-baseline-5w"

    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = f"checkpoints/{name}"

    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False  # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"  # Local directory for eval logs
    seed: int = 7

    action_decoder_path: str = f"checkpoints/{name}/action_decoder.pt"
    window_size: int = 30

    n_layers: int = 2
    hidden_dim: int = 1024

    with_proprio: bool = True
    wogripper: bool = True

    smooth: bool = False
    balancing_factor: float = 0.1  # larger for smoother

    task_name: str = "iros_pack_in_the_supermarket"


@draccus.wrap()
def get_policy(cfg: GenerateConfig) -> None:

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)

    return policy, cfg


if __name__ == "__main__":
    policy, cfg = get_policy()
    infer(policy, cfg)
