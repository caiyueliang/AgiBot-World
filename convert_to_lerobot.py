""" 
这个项目基于开源项目 🤗 LeRobot: https://github.com/huggingface/lerobot 

我们感谢 LeRobot 团队的杰出工作和对社区的贡献。

如果您觉得这个项目有用，请考虑支持和探索 LeRobot。
"""

import os
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy

import h5py
import torch
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pformat
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)

# 定义各种相机和传感器的键名常量
HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"
HEAD_CENTER_FISHEYE_COLOR = "head_center_fisheye_color.mp4"
HEAD_LEFT_FISHEYE_COLOR = "head_left_fisheye_color.mp4"
HEAD_RIGHT_FISHEYE_COLOR = "head_right_fisheye_color.mp4"
BACK_LEFT_FISHEYE_COLOR = "back_left_fisheye_color.mp4"
BACK_RIGHT_FISHEYE_COLOR = "back_right_fisheye_color.mp4"
HEAD_DEPTH = "head_depth"

# 默认图像存储路径格式
DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
)

# 定义数据集的特征结构，包括各种观测和动作的数据类型、形状等信息
FEATURES = {
    "observation.images.top_head": {  # 顶部头部相机图像
        "dtype": "video",  # 数据类型为视频
        "shape": [480, 640, 3],  # 图像形状：高度、宽度、通道
        "names": ["height", "width", "channel"],  # 维度名称
        "video_info": {  # 视频信息
            "video.fps": 30.0,  # 帧率
            "video.codec": "av1",  # 编码格式
            "video.pix_fmt": "yuv420p",  # 像素格式
            "video.is_depth_map": False,  # 不是深度图
            "has_audio": False,  # 没有音频
        },
    },
    "observation.images.cam_top_depth": {  # 顶部深度相机图像
        "dtype": "image",  # 数据类型为图像
        "shape": [480, 640, 1],  # 图像形状
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_left": {  # 左手相机图像
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.hand_right": {  # 右手相机图像
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_center_fisheye": {  # 头部中心鱼眼相机
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_left_fisheye": {  # 头部左侧鱼眼相机
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.head_right_fisheye": {  # 头部右侧鱼眼相机
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.back_left_fisheye": {  # 背部左侧鱼眼相机
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.images.back_right_fisheye": {  # 背部右侧鱼眼相机
        "dtype": "video",
        "shape": [748, 960, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": 30.0,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    },
    "observation.state": {  # 观测状态（机器人状态）
        "dtype": "float32",
        "shape": [20],  # 20维状态向量
    },
    "action": {  # 动作
        "dtype": "float32",
        "shape": [22],  # 22维动作向量
    },
    "episode_index": {  # 回合索引
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {  # 帧索引
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "index": {  # 全局索引
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "task_index": {  # 任务索引
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}


def get_stats_einops_patterns(dataset, num_workers=0):
    """这些einops模式将用于聚合批次并计算统计数据。

    注意：我们假设图像是通道优先格式
    """

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))  # 获取一个批次数据

    stats_patterns = {}  # 存储每种特征的统计模式

    for key in dataset.features:
        # 检查张量不是float64类型
        assert batch[key].dtype != torch.float64

        # 如果是相机图像数据
        if key in dataset.meta.camera_keys:
            # 检查图像是通道优先格式
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"期望通道优先图像，但得到 {batch[key].shape}"
            assert (
                batch[key].dtype == torch.float32
            ), f"期望 torch.float32，但得到 {batch[key].dtype=}"
            # 设置图像数据的统计模式
            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:  # 二维数据
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:  # 一维数据
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def compute_stats(dataset, batch_size=8, num_workers=4, max_num_samples=None):
    """计算LeRobotDataset中所有数据键的均值/标准差和最小值/最大值统计信息。"""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # 获取统计模式
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # 初始化统计量：均值、标准差、最大值、最小值
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        """创建带有固定种子的数据加载器，确保可重复性"""
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # 计算均值和最小/最大值
    first_batch = None  # 保存第一个批次用于验证
    running_item_count = 0  # 用于在线均值计算
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)  # 保存第一个批次
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # 计算当前批次的均值
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # 使用数值稳定的方式更新均值：x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            # 更新最大值和最小值
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    # 计算标准差
    first_batch_ = None
    running_item_count = 0  # 用于在线标准差计算
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # 检查批次顺序是否与之前相同
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # 计算平方残差的均值
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            # 更新标准差
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    # 计算最终的标准差（平方根）
    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    # 组织统计结果
    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats


class AgiBotDataset(LeRobotDataset):
    """AgiBot数据集类，继承自LeRobotDataset"""
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(
        self, task: str, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        我们重写此方法以将mp4视频复制到目标位置
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")  # 获取回合长度
        episode_index = episode_buffer["episode_index"]  # 获取回合索引
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): 添加使用现有episode_index的选项
            raise NotImplementedError(
                "您可能手动提供了episode_buffer，但其episode_index与数据集中的总回合数不匹配。目前不支持此操作。"
            )

        if episode_length == 0:
            raise ValueError(
                "在调用`add_episode`之前，必须使用`add_frame`添加一个或多个帧。"
            )

        task_index = self.meta.get_task_index(task)  # 获取任务索引

        print(f"[save_episode][episode_buffer.keys()] {episode_buffer.keys()}")
        print(f"[save_episode][self.features] {self.features}")
        # 👇 在这里移除 'task'，因为它不是 features 的一部分
        if 'task' in episode_buffer:
            episode_buffer.pop('task')

        # 检查缓冲区键是否与特征匹配
        if not set(episode_buffer.keys()) == set(self.features):
            raise ValueError()

        # 处理每个特征的数据
        for key, ft in self.features.items():
            if key == "index":
                # 生成全局索引
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                # 填充回合索引
                episode_buffer[key] = np.full((episode_length,), episode_index)
            elif key == "task_index":
                # 填充任务索引
                episode_buffer[key] = np.full((episode_length,), task_index)
            elif ft["dtype"] in ["image", "video"]:
                # 跳过图像和视频数据，它们已经单独处理
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                # 处理一维标量数据
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                # 处理一维向量数据
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()  # 等待图像写入完成
        self._save_episode_table(episode_buffer, episode_index)  # 保存回合数据表

        # 保存元数据
        self.meta.save_episode(episode_index, episode_length, task, task_index)
        
        # 处理视频文件
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)  # 创建目录
            shutil.copyfile(videos[key], video_path)  # 复制视频文件
            print(f"[save_episode][key] {key}, video_path: {video_path}, videos[key]: {videos[key]}")
        
        if not episode_data:  # 重置缓冲区
            self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False  # 标记为未整合

    def consolidate(
        self, run_compute_stats: bool = True, keep_image_files: bool = False
    ) -> None:
        """整合数据集，计算统计信息"""
        self.hf_dataset = self.load_hf_dataset()  # 加载HuggingFace数据集
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        # 检查时间戳同步
        check_timestamps_sync(
            self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
        )
        
        # 写入视频信息
        if len(self.meta.video_keys) > 0:
            self.meta.write_video_info()

        # 清理图像文件（如果不保留）
        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")
                
        # 验证视频文件数量
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        # 验证parquet文件数量
        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # 计算统计信息
        if run_compute_stats:
            self.stop_image_writer()  # 停止图像写入器
            self.meta.stats = compute_stats(self)  # 计算统计信息
            serialized_stats = serialize_dict(self.meta.stats)  # 序列化统计信息
            write_json(serialized_stats, self.root / STATS_PATH)  # 写入统计文件
            self.consolidated = True  # 标记为已整合
        else:
            logging.warning(
                "跳过数据集统计信息的计算，数据集未完全整合。"
            )

    def add_frame(self, frame: dict) -> None:
        """
        此函数仅将帧添加到episode_buffer中。除了图像（写入临时目录）之外，不会写入磁盘。
        要保存这些帧，需要调用'save_episode()'方法。
        """
        # TODO(aliberts, rcadene): 添加输入检查，检查是否为numpy或torch，检查dtype和shape是否匹配等。

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()  # 创建缓冲区

        frame_index = self.episode_buffer["size"]  # 获取当前帧索引
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )  # 获取或计算时间戳
        self.episode_buffer["frame_index"].append(frame_index)  # 添加帧索引
        self.episode_buffer["timestamp"].append(timestamp)  # 添加时间戳

        # 处理帧中的每个数据项
        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            # 将torch张量转换为numpy数组
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)  # 添加到缓冲区

        self.episode_buffer["size"] += 1  # 增加缓冲区大小


def load_depths(root_dir: str, camera_name: str):
    """加载深度图像数据"""
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))  # 获取所有深度图像文件
    # 加载并归一化深度图像（除以1000将毫米转换为米）
    return [np.array(Image.open(f)).astype(np.float32) / 1000 for f in all_imgs]


def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> list | None:
    """加载本地数据集并返回包含观测和动作的字典"""

    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"  # 观测数据目录
    depth_imgs = load_depths(ob_dir / "depth", HEAD_DEPTH)  # 加载深度图像
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"  # 本体感知数据目录

    # 从HDF5文件加载本体感知数据
    with h5py.File(proprio_dir / "proprio_stats.h5") as f:
        state_joint = np.array(f["state/joint/position"])  # 关节位置状态
        state_effector = np.array(f["state/effector/position"])  # 末端执行器位置状态
        state_head = np.array(f["state/head/position"])  # 头部位置状态
        state_waist = np.array(f["state/waist/position"])  # 腰部位置状态
        action_joint = np.array(f["action/joint/position"])  # 关节动作
        action_effector = np.array(f["action/effector/position"])  # 末端执行器动作
        action_head = np.array(f["action/head/position"])  # 头部动作
        action_waist = np.array(f["action/waist/position"])  # 腰部动作
        action_velocity = np.array(f["action/robot/velocity"])  # 机器人速度动作

    # 合并状态数据
    states_value = np.hstack(
        [state_joint, state_effector, state_head, state_waist]
    ).astype(np.float32)
    
    # 检查动作数据一致性
    assert (
        action_joint.shape[0] == action_effector.shape[0]
    ), f"action_joint形状:{action_joint.shape};action_effector形状:{action_effector.shape}"
    
    # 合并动作数据
    action_value = np.hstack(
        [action_joint, action_effector, action_head, action_waist, action_velocity]
    ).astype(np.float32)

    # 检查数据长度一致性
    assert len(depth_imgs) == len(
        states_value
    ), f"图像数量和状态数量不相等"
    assert len(depth_imgs) == len(
        action_value
    ), f"图像数量和动作数量不相等"
    
    # 构建帧数据列表
    frames = [
        {
            "observation.images.cam_top_depth": depth_imgs[i],  # 深度图像
            "observation.state": states_value[i],  # 状态
            "action": action_value[i],  # 动作
        }
        for i in range(len(depth_imgs))
    ]

    # 构建视频文件路径字典
    v_path = ob_dir / "videos"
    videos = {
        "observation.images.top_head": v_path / HEAD_COLOR,
        "observation.images.hand_left": v_path / HAND_LEFT_COLOR,
        "observation.images.hand_right": v_path / HAND_RIGHT_COLOR,
        "observation.images.head_center_fisheye": v_path / HEAD_CENTER_FISHEYE_COLOR,
        "observation.images.head_left_fisheye": v_path / HEAD_LEFT_FISHEYE_COLOR,
        "observation.images.head_right_fisheye": v_path / HEAD_RIGHT_FISHEYE_COLOR,
        "observation.images.back_left_fisheye": v_path / BACK_LEFT_FISHEYE_COLOR,
        "observation.images.back_right_fisheye": v_path / BACK_RIGHT_FISHEYE_COLOR,
    }
    return frames, videos


def get_task_instruction(task_json_path: str) -> dict:
    """获取任务语言指令"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)  # 加载任务信息
    task_name = task_info[0]["task_name"]  # 获取任务名称
    task_init_scene = task_info[0]["init_scene_text"]  # 获取初始场景描述
    task_instruction = f"{task_name}.{task_init_scene}"  # 组合任务指令
    print(f"获取任务指令 <{task_instruction}>")
    return task_instruction


def main(
    src_path: str,
    tgt_path: str,
    task_id: int,
    repo_id: str,
    task_info_json: str,
    debug: bool = False,
):
    """主函数：将原始数据转换为LeRobot格式的数据集"""
    task_name = get_task_instruction(task_info_json)  # 获取任务指令

    # 创建AgiBot数据集
    dataset = AgiBotDataset.create(
        repo_id=repo_id,
        root=f"{tgt_path}/{repo_id}",
        fps=30,  # 帧率
        robot_type="a2d",  # 机器人类型
        features=FEATURES,  # 特征定义
    )

    # 获取所有观测数据子目录
    all_subdir = sorted(
        [
            f.as_posix()
            for f in Path(src_path).glob(f"observations/{task_id}/*")
            if f.is_dir()
        ]
    )

    if debug:
        all_subdir = all_subdir[:2]  # 调试模式下只处理前2个目录

    # 获取所有回合ID
    all_subdir_eids = [int(Path(path).name) for path in all_subdir]

    # 加载所有本地数据集
    if debug:
        # 调试模式下顺序处理
        raw_datasets_before_filter = [
            load_local_dataset(subdir, src_path=src_path, task_id=task_id)
            for subdir in tqdm(all_subdir_eids)
        ]
    else:
        # 生产环境下使用多进程并行处理
        raw_datasets_before_filter = process_map(
            partial(load_local_dataset, src_path=src_path, task_id=task_id),
            all_subdir_eids,
            max_workers=os.cpu_count() // 2,  # 使用一半CPU核心
            desc="Generating local dataset",
        )
        
    # 移除结果为None的数据集
    raw_datasets = [
        dataset for dataset in raw_datasets_before_filter if dataset is not None
    ]

    # 移除对应子目录ID
    all_subdir_eids = [
        eid
        for eid, dataset in zip(all_subdir_eids, raw_datasets_before_filter)
        if dataset is not None
    ]
    
    # 生成任务描述列表
    all_subdir_episode_desc = [task_name] * len(all_subdir_eids)
    print(all_subdir_episode_desc)

    # 将原始数据添加到数据集中
    for raw_dataset, episode_desc in zip(
        tqdm(raw_datasets, desc="Generating dataset from raw datasets"),
        all_subdir_episode_desc,
    ):
        # 添加每个帧到数据集
        for raw_dataset_sub in tqdm(
            raw_dataset[0], desc="Generating dataset from raw dataset"
        ):
            dataset.add_frame(raw_dataset_sub)
        # 保存回合数据
        dataset.save_episode(task=episode_desc, videos=raw_dataset[1])
        
    # 整合数据集
    dataset.consolidate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
        help="源数据路径"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="任务ID"
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
        help="目标数据路径"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )
    args = parser.parse_args()  # 解析参数

    task_id = args.task_id
    json_file = f"{args.src_path}/task_info/task_{args.task_id}.json"  # 任务信息文件路径
    dataset_base = f"agibotworld/task_{args.task_id}"  # 数据集基础名称

    assert Path(json_file).exists, f"找不到 {json_file}。"  # 检查任务信息文件是否存在
    
    # 运行主函数
    main(args.src_path, args.tgt_path, task_id, dataset_base, json_file, args.debug)