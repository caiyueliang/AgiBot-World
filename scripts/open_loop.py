import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "InternVL"))
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Optional, Union
import draccus
import torch
import torch.distributed as dist
import tqdm
from PIL import Image
from transformers import AutoProcessor
import numpy as np
import matplotlib.pyplot as plt
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
import torch.distributed as dist
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
import prismatic.vla.datasets.pretrainAe_a2d_pretrain_v6 as a2d_cfg
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel


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


def calc_mse_for_single_trajectory(
    policy,
    dataset,
    cfg,
    traj_id: int,
    steps=300,
    action_horizon=30,
    plot=False,
):
    state_joints_across_time = []
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []

    length = len(dataset.data)
    steps = min(steps, length)

    for step in tqdm.tqdm(range(steps)):

        data_point = dataset.__getitem__(step)
        
        image = Image.open(os.path.join(dataset.data[step]["episode_dir"], "camera", str(step), "head_color.jpg"))
        lang = dataset.data[step]["detailed_job_description"]
        image_array = np.array(image)

        state = data_point["proprio"][0].numpy()
        gt_action = data_point["actions"][0].numpy()

        state_joints_across_time.append(state)
        gt_action_joints_across_time.append(gt_action)

        if cfg.with_proprio:
            action = policy.step(image_array, lang, state)
        else:
            action = policy.step(image_array, lang)

        concat_pred_action = action
        pred_action_joints_across_time.append(concat_pred_action)

    # plot the joints
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    assert (
        state_joints_across_time.shape
        == gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    )

    # calc MSE across time
    mse = np.mean((gt_action_joints_across_time - pred_action_joints_across_time) ** 2)
    print("Unnormalized Action MSE across single traj:", mse)

    num_of_joints = state_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))

        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory {traj_id}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            # ax.plot(state_joints_across_time[:, i], label="state joints")
            ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action joints")

            # put a dot every ACTION_HORIZON
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")

            ax.set_title(f"Joint {i}")
            ax.legend()

        plt.tight_layout()
        plt.savefig(cfg.save_path)

    return mse


@dataclass
class GenerateConfig:

    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/finetuned" 
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
                  
    center_crop: bool = False
    local_log_dir: str = "./experiments/eval_logs"
    seed: int = 7  

    action_decoder_path:str = "checkpoints/finetuned/action_decoder.pt"
    window_size: int = 30
    
    n_layers: int = 1
    hidden_dim: int = 512
    balancing_factor: float = 0.01                     # larger for smoother
    
    data_root_dir: str = ""  
    save_path: str = "openloop.png"
    
    with_proprio: bool = False
    
    debug: bool = False
    

@draccus.wrap()
def get_policy(cfg: GenerateConfig) -> None:

    setup_distributed()

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)
    

    # Load gensim dataset
    from prismatic.vla.datasets import A2dDataset
    dataset_args = a2d_cfg.DatasetArguments(
        meta_json_dir=cfg.data_root_dir,
        data_root_dir=cfg.data_root_dir,
    )
    data_training_args = a2d_cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = a2d_cfg.ActionSpacePadderArguments()

    text_tokenizer = AutoTokenizer.from_pretrained(
        "InternVL2-2B",
        trust_remote_code=True,
        add_eos_token=False,
    )

    text_tokenizer.model_max_length = 4096

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

    vla_dataset = A2dDataset(
        # base parmas
        label_file_dir=dataset_args.meta_json_dir, 
        data_root_dir=dataset_args.data_root_dir, 
        valid_episode_txt=dataset_args.valid_episode_txt, 
        world_size=dist.get_world_size(), 
        rank_id=dist.get_rank(), 
        sample_rate=dataset_args.train_sample_rate, 
        online_process_mp_cnt=dataset_args.online_process_mp_cnt, 
        # a2d params
        text_tokenizer=text_tokenizer, 
        num_image_token=int((dataset_args.force_image_size // 14) ** 2 * (0.5**2)), 
        is_train=True, 
        image_size=data_training_args.force_image_size, 
        pad2square=data_training_args.pad2square, 
        dynamic_image_size=data_training_args.dynamic_image_size, 
        use_thumbnail=data_training_args.use_thumbnail, 
        min_dynamic_patch=data_training_args.min_dynamic_patch, 
        max_dynamic_patch=data_training_args.max_dynamic_patch, 
        normalize_type=data_training_args.normalize_type, 
        action_chunk_size=data_training_args.action_chunk_size, 
        # use_real_state=data_training_args.use_real_state, 
        use_real_state=True, 
        conversation_type=data_training_args.conversation_type, 
        vis_frame=False, 
        vis_dir="", 
        ActionSpacePadder=ActionSpacePadder, 
        min_window_size=cfg.window_size, 
        max_window_size=cfg.window_size + 1, 
        image_transform=processor.image_processor.apply_transform, 
    )

    vla_dataset.generate_task_infos(
        dataset_args.dataset_task_cfg,
        task_episode_processors_cfg=dataset_args.episode_processors,
        task_dataset_processors_cfg=dataset_args.dataset_processors,
        task_runtime_processors_cfg=dataset_args.runtime_processors,
        shuffle=False,
        statistic=True,
        debug_one_episode=True,
    )

    return policy, vla_dataset, cfg


policy, dataset, cfg = get_policy()

mse = calc_mse_for_single_trajectory(
    policy,
    dataset,
    cfg,
    traj_id=0,
    steps=1000,
    action_horizon=30,
    plot=True
)

print("MSE loss for trajectory 0:", mse)