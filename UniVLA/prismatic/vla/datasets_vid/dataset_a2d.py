import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "InternVL"))
import copy
import json
import re
import random
import time

random.seed(42)
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import logging
logging.set_verbosity_info()
from prismatic.vla.datasets_vid.dataset_transforms import PipelineComposer
from prismatic.vla.datasets_vid.dataset_transforms import build_latent_image_transform
import sys
sys.path.append("/inspire/hdd/project/roboticsystem2/chenjin-CZXS24230112/AgiBot-World/UniVLA/InternVL")
from internvl_chat.internvl.train.dataset import build_transform, dynamic_preprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.get_logger("transformers.dataset_jaka" + __name__)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def timer(vis=True):
    """Timer."""

    def time_it(func):
        def inner(*arg, **kwarg):
            s_time = time.time()
            res = func(*arg, **kwarg)
            e_time = time.time()
            cost = e_time - s_time
            if vis:
                print(f"[{func.__qualname__}] cost time: {cost:.4f}")
                # logger.info("[%s] cost time: %.3f", func.__name__, cost)
            return res

        return inner

    return time_it


class MetaDataset(Dataset):
    def __init__(
        self,
        label_file_dir,
        data_root_dir,
        valid_episode_txt=None,
    ):
        self.label_file_dir = label_file_dir
        self.data_root_dir = data_root_dir
        if valid_episode_txt is None:
            self.valid_episodes = None
            logger.info(f"[DATASET] not Load valid_episode_txt")
        else:
            with open(valid_episode_txt, "r", encoding="utf-8") as fin:
                valid_episodes: List[str] = [line.strip() for line in fin]
            logger.info(f"[DATASET] Load {len(valid_episodes)} valid episode_ids from {valid_episode_txt}")
            self.valid_episodes = set(valid_episodes)

    def get_episode_path(self, episode_info, task_config):
        task_id = task_config["task_id"]
        evdir = os.path.join(
            self.data_root_dir,
            f'observations/{task_id}/{episode_info["episode_id"]}/videos',
        )
        espath = os.path.join(
            self.data_root_dir,
            f'proprio_stats/{task_id}/{episode_info["episode_id"]}/proprio_stats.h5',
        )
        return evdir, espath


class BaseDataset(MetaDataset):
    def __init__(
        self,
        label_file_dir=None,
        data_root_dir=None,
        world_size=None,
        rank_id=None,
        sample_rate=None,
        online_process_mp_cnt=1,
        valid_episode_txt=None,
    ):
        super().__init__(
            label_file_dir=label_file_dir, data_root_dir=data_root_dir, valid_episode_txt=valid_episode_txt
        )
        self.world_size = world_size
        self.rank_id = rank_id
        self.sample_rate = sample_rate
        self.online_process_mp_cnt = online_process_mp_cnt

        self.check_img = False
        self.check_corrupt = False

    @timer()
    def generate_task_infos(
        self,
        dataset_cfg,
        task_dataset_processors_cfg,
        task_runtime_processors_cfg,
        shuffle=True,
        statistic=False,
        debug_one_episode=False,
    ):
        self.dataset_cfg = dataset_cfg
        self.task_dataset_processors = PipelineComposer(task_dataset_processors_cfg)
        self.task_runtime_processors = PipelineComposer(task_runtime_processors_cfg)

        all_dataset_episode_info = []
        for idx, (task_id, task_config) in enumerate(dataset_cfg.items()):
            task_config["task_id"] = task_id
            label_file_name = os.path.join(self.label_file_dir, task_config["label_file_name"])
            with open(label_file_name, "r") as fid:
                label_list = json.load(fid)
            label_list = self.pack_addition_info(label_list, task_config)
            all_dataset_episode_info.extend(label_list)
            logger.info(f"label task{task_id} file: {label_file_name}, contains {len(label_list)} episode info.")

        # check_results, all_dataset_episode_info = self.episode_common_sanity_check_and_filter(all_dataset_episode_info)
        check_results, all_dataset_episode_info = self.episode_common_sanity_check_and_filter_parallel(all_dataset_episode_info)
        logger.info(f"sanity check: {check_results}")

        if debug_one_episode == True:
            logger.info(f"DEBUG MODE: Only use one episode!!")
            all_dataset_episode_info = all_dataset_episode_info[:1]

        logger.info(f"world_size: {self.world_size}, rank_id: {self.rank_id}")
        if self.world_size is None:
            sub_data_shard = all_dataset_episode_info
        else:
            random.shuffle(all_dataset_episode_info)  # shuffle before shard
            sub_data_shard = []
            for idx, info in enumerate(all_dataset_episode_info):
                if idx % self.world_size == self.rank_id:
                    sub_data_shard.append(info)

        self.raw_data = sub_data_shard
        logger.info(
            f"[rank:{self.rank_id}/worldsize:{self.world_size}] Get {len(self.raw_data)} episode from all {len(all_dataset_episode_info)} episode in {len(dataset_cfg)} dataset!"
        )
        self.data, self.data_infos = self._processor_pipeline_dataset(self.raw_data)

        if self.sample_rate is not None:
            data_sampled = []
            for idx, item in enumerate(self.data):
                if idx % self.sample_rate == 0:
                    data_sampled.append(item)
            del self.data
            self.data = None
            self.data = data_sampled
        logger.info(f"load {len(self.data)} pair data with sampling ratio: {self.sample_rate}")

        dist.barrier()
        original_length = len(self.data)
        shard_num = torch.tensor([original_length], dtype=torch.int64, device=torch.cuda.current_device())
        dist.all_reduce(shard_num, op=dist.ReduceOp.MIN, async_op=False)
        shard_num = shard_num.cpu().item()
        self.data = self.data[:shard_num]

        if shuffle:
            self.shuffle()
            logger.info(f"shuffle self.data")
        self.data_len = len(self.data)

        logger.info(f"Finally, get {len(self.data)} pair data, original len is {original_length}")

    def pack_addition_info(self, labels, task_config):
        for label in labels:
            label["evdir"], label["espath"] = self.get_episode_path(label, task_config)
            label["task_specific_cfg"] = task_config
            label["task_id"] = task_config["task_id"]
        return labels

    @timer()
    def _processor_pipeline_dataset(self, raw_data):
        data_infos = {}
        results = self.task_dataset_processors({"dataset": raw_data, "global_info": data_infos})
        data = results["iter_dataset"]
        return data, data_infos

    @timer()
    def episode_common_sanity_check_and_filter(self, episode_infos):
        sanity_check_result = {
            "evdir_not_exist": 0,
            "espath_not_exist": 0,
        }

        episode_info_filtered = []
        for ep_info in tqdm(episode_infos, desc="sanity_check", mininterval=60):
            evdir, espath = ep_info["evdir"], ep_info["espath"]
            if not os.path.exists(evdir):
                sanity_check_result["evdir_not_exist"] += 1
                continue

            if not os.path.exists(espath):
                sanity_check_result["espath_not_exist"] += 1
                continue

            episode_info_filtered.append(ep_info)
        return sanity_check_result, episode_info_filtered

    def episode_common_sanity_check_and_filter_parallel(self, episode_infos, num_workers=24):
        """并行版本的文件检查"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing as mp
        
        def check_episode_paths(ep_info):
            evdir, espath = ep_info["evdir"], ep_info["espath"]
            evdir_exists = os.path.exists(evdir)
            espath_exists = os.path.exists(espath)
            return ep_info, evdir_exists, espath_exists
        
        sanity_check_result = {
            "evdir_not_exist": 0,
            "espath_not_exist": 0,
        }
        
        episode_info_filtered = []
        
        # 使用线程池并行检查
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_episode = {
                executor.submit(check_episode_paths, ep_info): ep_info 
                for ep_info in episode_infos
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_episode), 
                            total=len(episode_infos), 
                            desc="sanity_check_parallel"):
                ep_info, evdir_exists, espath_exists = future.result()
                
                if not evdir_exists:
                    sanity_check_result["evdir_not_exist"] += 1
                    continue
                    
                if not espath_exists:
                    sanity_check_result["espath_not_exist"] += 1
                    continue
                    
                episode_info_filtered.append(ep_info)
        
        return sanity_check_result, episode_info_filtered

    def __len__(self):
        try:
            return len(self.data)
        except Exception as e:
            raise RuntimeError(f"task dataset may not init!, {e}")

    def __getitem__(self, idx):
        raw_target_ = self.data[idx % self.data_len]
        raw_target = copy.deepcopy(raw_target_)
        raw_target = self.task_runtime_processors(raw_target)
        return raw_target

    def shuffle(self):
        random.shuffle(self.data)


class A2dDataset(BaseDataset):
    def __init__(
        self,
        num_image_token,
        is_train=True,
        image_size=448,
        pad2square=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        normalize_type="imagenet",
        action_chunk_size=30,
        use_real_state=False,
        conversation_type=0,
        vis_frame=False,
        vis_dir=None,
        ActionSpacePadder=None,
        min_window_size: int = 16,
        max_window_size: int = 16,
        image_transform = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_image_token = num_image_token
        logger.info(f"[Dataset] num_image_token: {num_image_token}")
        logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
        logger.info(f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}")

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square

        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.action_chunk_size = action_chunk_size
        self.use_real_state = use_real_state
        self.conversation_type = conversation_type
        self.vis_frame = vis_frame
        self.vis_dir = vis_dir

        self.ActionSpacePadder = ActionSpacePadder

        self.min_window_size = min_window_size 
        self.max_window_size = max_window_size 

        self.image_transform = image_transform
        self.image_transform_lam = torchvision.transforms.ToTensor()
        self.resize_img = torchvision.transforms.Resize(224)

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    @staticmethod
    def make_conversation(conversation_type, prompt_dict):
        if conversation_type == 0:
            return f"What action should the robot take to {prompt_dict['job_description']}?"
        elif conversation_type == 1:
            random_num = random.random()
            if random_num < 0.33:
                return f"What action should the robot take to {prompt_dict['job_description']}?"
            elif random_num < 0.66:
                return f"The robot is performing the step of {prompt_dict['sub_job_description']}."
            else:
                return f"What action should the robot take to {prompt_dict['job_description']}? The robot is performing the step of {prompt_dict['sub_job_description']}."
        else:
            logger.error(f"Conversation Type {conversation_type} is not implemented.")
            raise NotImplementedError()

    def _get_conversation(self, raw_target):
        return A2dDataset.make_conversation(self.conversation_type, raw_target)

    def multi_image_get_item(
        self,
        raw_target: Dict[str, Any],
        cam_keys: List[str] = ["cam_tensor_head_color", "cam_tensor_hand_right_color", "cam_tensor_hand_left_color"],
    ):

        images, num_tiles = [], []
        num_image = 0

        ret = {}
        for cam_key in cam_keys:
            if cam_key in raw_target:
                num_image += 1
                if self.dynamic_image_size:
                    image = dynamic_preprocess(
                        raw_target[cam_key],
                        min_num=self.min_dynamic_patch,
                        max_num=self.max_dynamic_patch,
                        image_size=self.image_size,
                        use_thumbnail=self.use_thumbnail,
                    )
                    images += image
                    num_tiles.append(len(image))
                else:
                    if "hist" in cam_key:
                        ret[cam_key[:10]+"pixel_values"] = self.image_transform_lam(self.resize_img(raw_target[cam_key]))
                    else:
                        if "init" in cam_key:
                            if "head" in cam_key:
                                ret["head_pixel_values"] = self.image_transform(raw_target[cam_key])
                                ret[cam_key[:5]+"pixel_values"] = self.image_transform_lam(self.resize_img(raw_target[cam_key]))   
                            if "left" in cam_key:
                                ret["left_hand_pixel_values"] = self.image_transform(raw_target[cam_key])
                            if "right" in cam_key:
                                ret["right_hand_pixel_values"] = self.image_transform(raw_target[cam_key])                 
                        else:
                            ret[cam_key[:5]+"pixel_values"] = self.image_transform_lam(self.resize_img(raw_target[cam_key]))   
                                     
        ret["pixel_values"] = torch.cat((ret["head_pixel_values"], ret["left_hand_pixel_values"], ret["right_hand_pixel_values"]), dim=0)
        
        return ret
    

    def __getitem__(self, idx):
        get_data_done = False
        while not get_data_done:
            try:
                raw_target = super().__getitem__(idx)
                
                task_id = raw_target["task_id"]
                job_id = raw_target["job_id"]
                sn_code = raw_target["sn_code"]
                episode_id = raw_target["episode_id"]
                frame_idx = raw_target["frame_idx"]
                
                window_size = raw_target["window_size"]

                action, action_mask = self.ActionSpacePadder.get_action(raw_target["action_target"], chunk_size=self.action_chunk_size)
                state, state_mask = self.ActionSpacePadder.get_action(raw_target["agent_state"], chunk_size=1)

                results = self.multi_image_get_item(
                    raw_target, 
                    cam_keys=[
                        "init_cam_tensor_head_color", 
                        "init_cam_tensor_hand_right_color", 
                        "init_cam_tensor_hand_left_color", 
                        "goal_cam_tensor_head_color", 
                        "hist_init_cam_tensor_head_color", 
                        "hist_goal_cam_tensor_head_color", 
                        ]
                        )
                freq = int(raw_target["used_cam_cfg"]["head"]["camera_fps"])
                agent_state = torch.tensor(state, dtype=torch.float32)
                if not self.use_real_state:
                    agent_state = -1 * torch.ones_like(agent_state)
                
                results.update(
                    {
                        "actions": torch.tensor(action, dtype=torch.float32),
                        "actions_mask": None,
                        "proprio": agent_state,
                        "ctrl_freqs": torch.tensor([freq], dtype=torch.float32),
                        "window_size": window_size,
                        "lang": raw_target["detailed_job_description"], 
                        "task_id": task_id,
                        "job_id": job_id,
                        "sn_code": sn_code,
                        "episode_id": episode_id,
                        "frame_idx": frame_idx,
                    }
                )
                
                get_data_done = True
            except Exception as error:
                logger.error(f"process dataset idx: {idx}, {self.data[idx % self.data_len]['episode_dir']}, error info: {error}")
                idx = random.randint(0, len(self.data) - 1)

        return results


class EgoDataset(MetaDataset):
    def __init__(
        self,
        label_file_dir=None,
        data_root_dir=None,
        world_size=None,
        rank_id=None,
        sample_rate=None,
        online_process_mp_cnt=1,
        valid_episode_txt=None,
    ):
        super().__init__(
            label_file_dir=label_file_dir, data_root_dir=data_root_dir, valid_episode_txt=valid_episode_txt
        )
        self.world_size = world_size
        self.rank_id = rank_id
        self.sample_rate = sample_rate
        self.online_process_mp_cnt = online_process_mp_cnt

        self.check_img = False
        self.check_corrupt = False

    @timer()
    def generate_task_infos(
        self,
        dataset_cfg,
        task_dataset_processors_cfg,
        task_runtime_processors_cfg,
        shuffle=True,
        statistic=False,
        debug_one_episode=False,
    ):
        self.dataset_cfg = dataset_cfg
        self.task_dataset_processors = PipelineComposer(task_dataset_processors_cfg)
        self.task_runtime_processors = PipelineComposer(task_runtime_processors_cfg)

        all_dataset_episode_info = []
        for idx, (task_id, task_config) in enumerate(dataset_cfg.items()):
            task_config["task_id"] = task_id
            label_file_name = os.path.join(self.label_file_dir, task_config["label_file_name"])
            with open(label_file_name, "r") as fid:
                label_list = json.load(fid)
            label_list = self.pack_addition_info(label_list, task_config)
            all_dataset_episode_info.extend(label_list)
            logger.info(f"label task{task_id} file: {label_file_name}, contains {len(label_list)} episode info.")

        check_results, all_dataset_episode_info = self.episode_common_sanity_check_and_filter_parallel(all_dataset_episode_info)
        logger.info(f"sanity check: {check_results}")

        if debug_one_episode == True:
            logger.info(f"DEBUG MODE: Only use one episode!!")
            all_dataset_episode_info = all_dataset_episode_info[:1]

        logger.info(f"world_size: {self.world_size}, rank_id: {self.rank_id}")
        if self.world_size is None:
            sub_data_shard = all_dataset_episode_info
        else:
            random.shuffle(all_dataset_episode_info)  # shuffle before shard
            sub_data_shard = []
            for idx, info in enumerate(all_dataset_episode_info):
                if idx % self.world_size == self.rank_id:
                    sub_data_shard.append(info)

        self.raw_data = sub_data_shard
        logger.info(
            f"[rank:{self.rank_id}/worldsize:{self.world_size}] Get {len(self.raw_data)} episode from all {len(all_dataset_episode_info)} episode in {len(dataset_cfg)} dataset!"
        )
        self.data, self.data_infos = self._processor_pipeline_dataset(self.raw_data)

        if self.sample_rate is not None:
            data_sampled = []
            for idx, item in enumerate(self.data):
                if idx % self.sample_rate == 0:
                    data_sampled.append(item)
            del self.data
            self.data = None
            self.data = data_sampled
        logger.info(f"load {len(self.data)} pair data with sampling ratio: {self.sample_rate}")

        dist.barrier()
        original_length = len(self.data)
        shard_num = torch.tensor([original_length], dtype=torch.int64, device=torch.cuda.current_device())
        dist.all_reduce(shard_num, op=dist.ReduceOp.MIN, async_op=False)
        shard_num = shard_num.cpu().item()
        self.data = self.data[:shard_num]

        if shuffle:
            self.shuffle()
            logger.info(f"shuffle self.data")
        self.data_len = len(self.data)

        logger.info(f"Finally, get {len(self.data)} pair data, original len is {original_length}")

    def pack_addition_info(self, labels, task_config):
        for label in labels:
            label["evdir"], label["espath"] = self.get_episode_path(label, task_config)
            label["task_specific_cfg"] = task_config
            label["task_id"] = task_config["task_id"]
        return labels

    @timer()
    def _processor_pipeline_dataset(self, raw_data):
        data_infos = {}
        results = self.task_dataset_processors({"dataset": raw_data, "global_info": data_infos})
        data = results["iter_dataset"]
        return data, data_infos

    @timer()
    def episode_common_sanity_check_and_filter_parallel(self, episode_infos, num_workers=24):
        """并行版本的文件检查"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing as mp
        
        def check_episode_paths(ep_info):
            evdir = ep_info["evdir"]
            evdir_exists = os.path.exists(evdir)
            return ep_info, evdir_exists
        
        sanity_check_result = {
            "evdir_not_exist": 0,
        }
        
        episode_info_filtered = []
        
        # 使用线程池并行检查
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_episode = {
                executor.submit(check_episode_paths, ep_info): ep_info 
                for ep_info in episode_infos
            }
            
            # 收集结果
            for future in tqdm(as_completed(future_to_episode), 
                            total=len(episode_infos), 
                            desc="sanity_check_parallel"):
                ep_info, evdir_exists = future.result()
                
                if not evdir_exists:
                    sanity_check_result["evdir_not_exist"] += 1
                    continue
                    
                episode_info_filtered.append(ep_info)
        
        return sanity_check_result, episode_info_filtered

    def __len__(self):
        try:
            return len(self.data)
        except Exception as e:
            raise RuntimeError(f"task dataset may not init!, {e}")

    def __getitem__(self, idx):
        raw_target_ = self.data[idx % self.data_len]
        raw_target = copy.deepcopy(raw_target_)
        raw_target = self.task_runtime_processors(raw_target)
        return raw_target

    def shuffle(self):
        random.shuffle(self.data)


class LAMStage1Dataset(EgoDataset):
    def __init__(self, is_train=True, image_size=448, pad2square=False, normalize_type="imagenet", **kwargs):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.normalize_type = normalize_type

                    
    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def __getitem__(self, idx):
        get_data_done = False
        while not get_data_done:
            try:
                raw_target = super().__getitem__(idx)
                results = {}
                freq = 30

                results.update(
                    {
                        "random_video_len": raw_target["random_video_len"],
                        "videos": raw_target["videos"],
                        "ctrl_freqs": torch.tensor([freq], dtype=torch.float32),
                    }
                )

                get_data_done = True
            except Exception as error:
                logger.error(f"process dataset idx: {idx}, {self.data[idx % self.data_len]}, error info: {error}")
                idx = random.randint(0, len(self.data) - 1)

        return results
    
    
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
        os.environ["MASTER_PORT"] = "12345"
    # Parse environment variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the distributed environment
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)  # or 'gloo' for CPU

    # Set device for this process
    torch.cuda.set_device(local_rank)

    return local_rank, world_size
