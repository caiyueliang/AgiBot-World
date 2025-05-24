import math
from os import listdir, makedirs, path
from random import choices, randint
from typing import Any, Callable, Dict

import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info



def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 16,
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            val_batch_size: int = None,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )


class Platformer2D(Dataset):
    def __init__(
            self,
            split_path: str,
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            num_frames: int = 16,
            output_format: str = "t h w c"
    ) -> None:
        super(Platformer2D, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.num_frames = num_frames
        self.output_format = output_format

        # Get all the file path based on the split path
        self.file_names = []
        for file_name in listdir(split_path):
            self.file_names.append(path.join(split_path, file_name))

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
            self,
            video_path: str,
            num_frames: int,
            start_frame: int = None,
            frame_skip: int = 1
    ) -> Tensor:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if "retro" in video_path:
            frame_skip = 4
        elif "libero" in video_path or "openx" in video_path or "ego4d" in video_path:
            frame_skip = 2
        num_frames = num_frames * frame_skip

        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - num_frames))
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video, deal with padding and return
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break
        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        # Pad the video to be square
        if video.shape[1] != video.shape[2]:
            square_len = max(video.shape[1], video.shape[2])
            h_pad = (square_len - video.shape[1]) // 2
            w_pad = (square_len - video.shape[2]) // 2
            video = F.pad(video, (0, 0, w_pad, w_pad, h_pad, h_pad), value=0)

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> c t h w")
            video = F.interpolate(video, size=(self.resolution, self.resolution), mode="nearest")
            video = rearrange(video, f"c t h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        data_dict = {
            "videos": video
        }
        return data_dict


class OriginalPlatformer2D(Dataset):
    def __init__(
            self,
            data_root: str,
            env_source: str = "libero",
            split: str = "train",
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            num_frames: int = 16,
            output_format: str = "t h w c",
            alive: bool = False
    ) -> None:
        super(OriginalPlatformer2D, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.num_frames = num_frames
        self.output_format = output_format
        postfix = "_alive" if alive else ""

        # Get all the file path based on the split
        folders = []
        if env_source == "libero":
            folders.append(path.join(data_root, "libero", split))
        elif env_source == "openx":
            for env in listdir(path.join(data_root, "openx")):
                folders.append(path.join(data_root, "openx", env, split))
        elif env_source == "ego4d":
            folders.append(path.join(data_root, "ego4d", split))
        elif env_source == "embodied":
            folders.append(path.join(data_root, "ego4d", split))
            for env in listdir(path.join(data_root, "openx")):
                folders.append(path.join(data_root, "openx", env, split))
        elif env_source == "procgen":
            for env in listdir(path.join(data_root, f"procgen{postfix}")):
                folders.append(path.join(data_root, f"procgen{postfix}", env, split))
        elif env_source == "retro":
            for env in listdir(path.join(data_root, f"retro{postfix}")):
                folders.append(path.join(data_root, f"retro{postfix}", env, split))
        elif env_source == "both":
            for env in listdir(path.join(data_root, f"procgen{postfix}")):
                folders.append(path.join(data_root, f"procgen{postfix}", env, split))
            for env in listdir(path.join(data_root, f"retro{postfix}")):
                folders.append(path.join(data_root, f"retro{postfix}", env, split))
        elif path.exists(path.join(data_root, f"procgen{postfix}", env_source, split)):
            folders.append(path.join(data_root, f"procgen{postfix}", env_source, split))
        elif path.exists(path.join(data_root, f"retro{postfix}", env_source, split)):
            folders.append(path.join(data_root, f"retro{postfix}", env_source, split))
        else:
            raise ValueError(f"Invalid source: {env_source}")
        self.file_names = []
        for folder in folders:
            self.file_names.extend([
                path.join(folder, f)
                for f in listdir(folder)
            ])

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        while True:
            try:
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0
                )
                return self.build_data_dict(video)
            except:
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
            self,
            video_path: str,
            num_frames: int,
            start_frame: int = None,
            frame_skip: int = 1
    ) -> Tensor:
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if "retro" in video_path:
            frame_skip = 4
        elif "libero" in video_path or "openx" in video_path or "ego4d" in video_path:
            frame_skip = 2
        num_frames = num_frames * frame_skip

        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - num_frames))
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # Frame was successfully read, parse it
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                # Reach the end of video, deal with padding and return
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                else:
                    raise ValueError(f"Invalid padding type: {self.padding}")
                break
        cap.release()

        video = torch.stack(frames[::frame_skip]) / 255.0

        # Pad the video to be square
        if video.shape[1] != video.shape[2]:
            square_len = max(video.shape[1], video.shape[2])
            h_pad = (square_len - video.shape[1]) // 2
            w_pad = (square_len - video.shape[2]) // 2
            video = F.pad(video, (0, 0, w_pad, w_pad, h_pad, h_pad), value=0)

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> c t h w")
            video = F.interpolate(video, size=(self.resolution, self.resolution), mode="nearest")
            video = rearrange(video, f"c t h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        data_dict = {
            "videos": video
        }
        return data_dict


class MultiSourceSamplerDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            env_source: str = "libero",
            split: str = "train",
            samples_per_epoch: int = 1000000,
            sampling_strategy: str = "sample",
            alive: bool = False,
            **kwargs
    ) -> None:
        self.samples_per_epoch = samples_per_epoch
        postfix = "_alive" if alive else ""

        # Create all subsets
        folders = []
        if env_source == "libero":
            folders.append(path.join(data_root, "libero", split))
        elif env_source == "openx":
            for env in listdir(path.join(data_root, "openx")):
                folders.append(path.join(data_root, "openx", env, split))
        elif env_source == "ego4d":
            folders.append(path.join(data_root, "ego4d", split))
        elif env_source == "embodied":
            folders.append(path.join(data_root, "ego4d", split))
            for env in listdir(path.join(data_root, "openx")):
                folders.append(path.join(data_root, "openx", env, split))
        elif env_source == "procgen":
            for env in listdir(path.join(data_root, f"procgen{postfix}")):
                folders.append(path.join(data_root, f"procgen{postfix}", env, split))
        elif env_source == "retro":
            for env in listdir(path.join(data_root, f"retro{postfix}")):
                folders.append(path.join(data_root, f"retro{postfix}", env, split))
        elif env_source == "both":
            for env in listdir(path.join(data_root, f"procgen{postfix}")):
                folders.append(path.join(data_root, f"procgen{postfix}", env, split))
            for env in listdir(path.join(data_root, f"retro{postfix}")):
                folders.append(path.join(data_root, f"retro{postfix}", env, split))
        elif path.exists(path.join(data_root, f"procgen{postfix}", env_source, split)):
            folders.append(path.join(data_root, f"procgen{postfix}", env_source, split))
        elif path.exists(path.join(data_root, f"retro{postfix}", env_source, split)):
            folders.append(path.join(data_root, f"retro{postfix}", env_source, split))
        else:
            raise ValueError(f"Invalid source: {env_source}")
        self.subsets = []
        for folder in folders:
            self.subsets.append(Platformer2D(split_path=folder, **kwargs))
        # TODO: ablation as self.subsets = self.subsets[::10]
        print("Number of subsets:", len(self.subsets))

        if sampling_strategy == "sample":
            # Sample uniformly from all samples
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            # Sample uniformly from all datasets
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "log":
            # Generate probabilities according to the scale of each dataset
            probs = [math.log(len(d)) if len(d) else 0 for d in self.subsets]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        total_prob = sum(probs)
        assert total_prob > 0, "No sample is available"
        self.sample_probs = [x / total_prob for x in probs]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
        index (int): Index (ignored since we sample randomly).

        Returns:
        TensorDict: Dict containing all the data blocks.
        """

        # Randomly select a subset based on weights
        subset = choices(self.subsets, self.sample_probs)[0]

        # Sample a valid sample with a random index
        sample_idx = randint(0, len(subset) - 1)
        sample_item = subset[sample_idx]
        return sample_item


class LightningPlatformer2D(LightningDataset):
    """
    This dataset samples video recorded using a random agent
    playing the gym environments defined in the Procgen Benchmark,
    see Cobbe et al. ICML (2020).
    """

    def __init__(
            self,
            data_root: str,
            env_source: str = "libero",
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            num_frames: int = 16,
            output_format: str = "t h w c",
            samples_per_epoch: int = 1000000,
            sampling_strategy: str = "sample",
            alive: bool = False,
            make_data_pair: bool = False,
            **kwargs
    ) -> None:
        super(LightningPlatformer2D, self).__init__(**kwargs)
        self.data_root = data_root
        self.env_source = env_source
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.num_frames = num_frames
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy
        self.alive = alive
        self.make_data_pair = make_data_pair

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = MultiSourceSamplerDataset(
                data_root=self.data_root,
                env_source=self.env_source,
                split="train",
                padding=self.padding,
                randomize=self.randomize,
                resolution=self.resolution,
                num_frames=self.num_frames,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch,
                sampling_strategy=self.sampling_strategy,
                alive=self.alive
            )
            self.val_dataset = MultiSourceSamplerDataset(
                data_root=self.data_root,
                env_source=self.env_source,
                split="val",
                padding=self.padding,
                randomize=self.randomize,
                resolution=self.resolution,
                num_frames=self.num_frames,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch // 100,
                sampling_strategy=self.sampling_strategy,
                alive=self.alive
            )
        elif stage == "test":
            if self.make_data_pair:
                makedirs("output_pairs", exist_ok=True)
                completed = len(listdir("output_pairs"))
                todo_name = listdir(path.join(self.data_root, self.env_source))[completed]
                self.test_dataset = OriginalPlatformer2D(
                    data_root=self.data_root,
                    env_source=todo_name,
                    split="train",
                    padding=self.padding,
                    randomize=self.randomize,
                    resolution=self.resolution,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    alive=self.alive
                )
            else:
                self.test_dataset = OriginalPlatformer2D(
                    data_root=self.data_root,
                    env_source=self.env_source,
                    split="test",
                    padding=self.padding,
                    randomize=self.randomize,
                    resolution=self.resolution,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    alive=self.alive
                )
        else:
            raise ValueError(f"Invalid stage: {stage}")


