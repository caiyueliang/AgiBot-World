import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# import fnmatch
import json
import re
import traceback
from functools import lru_cache

import cv2
import h5py
import numpy as np
import yaml

from configs.state_vec import STATE_VEC_IDX_MAPPING

META_DIR = "dataset/iros-sim/multitask"

NORM_GRIPPER = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 120, 120]


@lru_cache(maxsize=4096)
def cache_read_h5(h5path):
    f = h5py.File(os.path.join(h5path, "aligned_joints.h5"))
    joint = np.array(f["state"]["joint"]["position"])
    try:
        gripper = np.array(f["state"]["effector"]["position"])
    except:
        gripper_l = np.array(f["state"]["left_effector"]["position"])
        gripper_r = np.array(f["state"]["right_effector"]["position"])
        if len(gripper_l.shape) == 1:
            gripper_l = np.expand_dims(gripper_l, axis=-1)
        if len(gripper_r.shape) == 1:
            gripper_r = np.expand_dims(gripper_r, axis=-1)
        gripper = np.concatenate((gripper_l, gripper_r), axis=1)
        
    return (
        joint,
        gripper,
    )


class HDF5A2DDataset:
    """
    This class is used to sample episodes from the A2D dataset
    stored in HDF5.
    """

    def __init__(self, use_tasks=None, save_path="./data/awsim") -> None:
        # Specify the tasks to use or use all tasks found in the meta data (json) directory
        self.all_tasks_dict = self.get_all_task_dict(META_DIR)
        
        self.DATASET_NAME = "aw_sim"
        use_tasks = self.all_tasks_dict.keys()

        self.save_path = save_path

        # Load the config
        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]

        self.file_paths = {}  # {task_id: [file_paths]}
        self.filepath2staticframe = {}
        self.index2epi = []
        self.epi2meta = {}

        for t in use_tasks:
            self.file_paths[t] = []

            data_prefix = f"{save_path}/{t}_"

            # if the npy file is not stored before
            if not os.path.exists(f"{data_prefix}filepath2staticframe.npy"):
                # js = json.load(open(meta_path))
                js = json.load(open(os.path.join(META_DIR, f"task_{t}_train.json")))
                file_paths = []
                index2epi = []
                epi2meta = {}
                # go through each episode
                for meta in js:
                    episode_id = meta.get("episode_id", None)
                    h5path = os.path.join(META_DIR, str(meta["task_id"]), str(meta["job_id"]), meta["sn_code"], str(meta["episode_id"]))
                    if os.path.exists(os.path.join(h5path, "aligned_joints.h5")):
                        try:
                            metajs = json.load(open(os.path.join(h5path, "meta_info.json")))
                            if metajs["version"] == "v0.0.2":
                                valid, res = self.parse_hdf5_file_state_only(h5path)
                                if valid:
                                    file_paths.append(h5path)
                                    index2epi.append(episode_id)
                                    epi2meta[episode_id] = {
                                        "english_task_name": meta.get("task_name", None),
                                        "episode_id": episode_id,
                                        "label_info": meta.get("label_info", None),
                                    }
                                else:
                                    print(f"Failed to parse {h5path}")
                        except Exception:
                            print(traceback.format_exc())
                    else:
                        print(f"File not found: {h5path}")

                np.save(f"{data_prefix}file_paths.npy", file_paths)
                np.save(f"{data_prefix}index2epi.npy", index2epi)
                np.save(f"{data_prefix}epi2meta.npy", epi2meta)
                np.save(f"{data_prefix}filepath2staticframe.npy", self.filepath2staticframe)

                self.file_paths[t].extend(file_paths)
                self.index2epi.extend(index2epi)
                self.epi2meta.update(epi2meta)
            else:
                self.file_paths[t].extend(np.load(f"{data_prefix}file_paths.npy", allow_pickle=True))
                self.index2epi.extend(np.load(f"{data_prefix}index2epi.npy", allow_pickle=True))
                self.epi2meta.update(np.load(f"{data_prefix}epi2meta.npy", allow_pickle=True).item())
                self.filepath2staticframe.update(
                    np.load(f"{data_prefix}filepath2staticframe.npy", allow_pickle=True).item()
                )

        self.total_episodes = 0
        self.task_sample_weights = []
        self.file_paths_indices = {}
        self.sum_task_lens = {}
        for k, v in self.file_paths.items():
            self.task_sample_weights.append(len(v))
            self.file_paths_indices[k] = list(range(len(v)))
            self.sum_task_lens[k] = self.total_episodes  # this task starts from this index of episode
            self.total_episodes += len(v)
        self.task_sample_weights /= np.sum(self.task_sample_weights)
        print(f"Total {self.total_episodes} episodes found.")
        assert self.total_episodes == len(self.epi2meta)

    def __len__(self):
        return self.total_episodes * 1000

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, task_name=None, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            # Select a task
            if task_name is None:
                task_name = np.random.choice(list(self.file_paths.keys()), p=self.task_sample_weights)

            # Select an episode
            if index is None:
                if len(self.file_paths_indices[task_name]) == 0:
                    continue
                index = np.random.choice(self.file_paths_indices[task_name])
            file_path = self.file_paths[task_name][index]

            episode_meta = self.epi2meta[self.index2epi[index + self.sum_task_lens[task_name]]]
            valid, sample = (
                self.parse_hdf5_file(file_path, episode_meta, task_name)
                if not state_only
                else self.parse_hdf5_file_state_only(file_path)
            )
            if valid:
                return sample
            else:
                # If the episode is invalid, we resample
                index = None

    def parse_hdf5_file(self, file_path, episode_meta, task_name, step_id=None):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(os.path.join(file_path, "aligned_joints.h5"), "r") as f:
            st, end, _len = self.filepath2staticframe[file_path]
            joints = f["state"]["joint"]["position"]
            num_steps = joints.shape[0]

            # Get meta info
            # We randomly sample a timestep
            if step_id is None:
                step_id = np.random.randint(st, end - 1)  # [0, num_step - 2]

            instruction = "".join([step["english_action_text"] for step in episode_meta["label_info"]["action_config"]])
            # instruction = os.path.join(self.save_path, f"task_{task_name}.pt")
            # assert os.path.exists(instruction)

            episode_id = episode_meta["episode_id"]

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction,
                "episode_id": episode_id,
            }

            # Parse the state and action
            # state_gripper = f["state"]["effector"]["position"]
            state_l_gripper = f["state"]["left_effector"]["position"]
            state_r_gripper = f["state"]["right_effector"]["position"]
            if len(state_l_gripper.shape) == 1:
                state_l_gripper = np.expand_dims(state_l_gripper, axis=-1)
            if len(state_r_gripper.shape) == 1:
                state_r_gripper = np.expand_dims(state_r_gripper, axis=-1)
            state_gripper = np.concatenate((state_l_gripper, state_r_gripper), axis=-1)
            
            state_joints = joints
            # Rescale gripper to [0, 1]
            state = np.concatenate([state_joints, state_gripper], axis=-1) / np.array([NORM_GRIPPER])
            # Parse the state and action
            state_std = np.std(state, axis=0)
            state_mean = np.mean(state, axis=0)
            state_norm = np.sqrt(np.mean(state**2, axis=0))
            state = state[step_id : step_id + 1]

            action_l_gripper = f["state"]["left_effector"]["position"][step_id + 1 : step_id + self.CHUNK_SIZE + 1]
            action_r_gripper = f["state"]["right_effector"]["position"][step_id + 1 : step_id + self.CHUNK_SIZE + 1]
            if len(action_l_gripper.shape) == 1:
                action_l_gripper = np.expand_dims(action_l_gripper, axis=-1)
            if len(action_r_gripper.shape) == 1:
                action_r_gripper = np.expand_dims(action_r_gripper, axis=-1)
            action_gripper = np.concatenate((action_l_gripper, action_r_gripper), axis=-1)
            action_joints = joints[step_id + 1 : step_id + self.CHUNK_SIZE + 1]
            # Rescale gripper to [0, 1]
            actions = np.concatenate([action_joints, action_gripper], axis=-1) / np.array([NORM_GRIPPER])

            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate(
                    [actions, np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))], axis=0
                )

            # Fill the state/action into the unified vector
            state = self.fill_in_state(state)
            state_indicator = self.fill_in_state(np.ones_like(state_std))
            state_std = self.fill_in_state(state_std)
            state_mean = self.fill_in_state(state_mean)
            state_norm = self.fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = self.fill_in_state(actions)

            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    img = os.path.join(file_path, "camera", str(i), key + ".jpg")
                    imgs.append(cv2.imread(img, cv2.IMREAD_COLOR))
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate(
                        [np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)), imgs], axis=0
                    )
                return imgs

            # `cam_high` is the external camera image
            cam_high = parse_img("head_color")
            # For step_id = st, the valid_len should be one
            valid_len = min(step_id - st + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
            cam_left_wrist = parse_img("hand_left_color")
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img("hand_right_color")
            cam_right_wrist_mask = cam_high_mask.copy()

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }

    def parse_hdf5_file_state_only(self, file_path, EPS=np.pi / 180 / 30):
        """[Modify] Parse a hdf5 file to generate a state trajectory.
        Full trajectory, not related to CHUNK_SIZE

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        joints, gripper = cache_read_h5(file_path)
        # filter dex hand data
        # if gripper.shape[1] == 12:
        #     return False, None

        qpos = np.concatenate([joints[:], gripper[:]], axis=-1)
        num_steps = qpos.shape[0]

        # [Optional] We skip the first few still steps
        # Get the idx of the first qpos whose delta exceeds the threshold
        qpos_delta = np.abs(qpos[:, :14] - qpos[0:1, :14])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError(f"Found no qpos that exceeds the threshold in {file_path}.")

        # Get the idx of the last qpos after which the robot is statis
        qpos_delta = np.abs(qpos[:, :14] - qpos[-1:, :14])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            last_idx = indices[-1]
        else:
            raise ValueError(f"Found no qpos that exceeds the threshold in {file_path}.")

        st = first_idx - 1
        end = min(num_steps, last_idx + 2)
        # [first_idx-1, last_idx+1] = [st, end-1]
        self.filepath2staticframe[file_path] = (st, end, end - st - 1)

        # Rescale gripper to [0, 1]
        qpos = qpos[st:end] / np.array([NORM_GRIPPER])

        # [Optional] We drop too-short episode
        num_steps = qpos.shape[0]
        if num_steps < 128:
            return False, None

        target_qpos = qpos[1:]
        qpos = qpos[:-1]

        # Parse the state and action
        state = self.fill_in_state(qpos)  # [st:end-1]
        action = self.fill_in_state(target_qpos)  # [st+1:end]

        # Return the resulting sample
        return True, {"state": state, "action": action}

    def get_all_task_dict(self, meta_data_dir):
        task_dict = {}
        for file in os.listdir(meta_data_dir):
            if file.endswith(".json"):
                task_id = re.search(r"task_(\d+)_train\.json", file).group(1)
                # if int(task_id) in [6]:
                task_dict[str(task_id)] = {
                    "meta_path": file,
                }
        return task_dict

    def get_full_trajectory(self, task_name=None, index: int = None, interval: int = 64):
        if task_name is None:
            task_name = np.random.choice(list(self.file_paths.keys()), p=self.task_sample_weights)

        if index is None:
            index = np.random.choice(self.file_paths_indices[task_name])
        file_path = self.file_paths[task_name][index]
        episode_meta = self.epi2meta[self.index2epi[index + self.sum_task_lens[task_name]]]
        trajectory = []
        st, end, _len = self.filepath2staticframe[file_path]
        # print(f"Episode length: {_len}")

        for i in range(st, end, interval):
            valid, sample = self.parse_hdf5_file(file_path, episode_meta, task_name, step_id=i)
            if valid:
                trajectory.append(sample)

        return trajectory

    def fill_in_state(self, values):
        # Target indices corresponding to your state space
        # In this example: 6 joints + 1 gripper for each arm
        UNI_STATE_INDICES = (
            [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(7)]
            + [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)]
            + [STATE_VEC_IDX_MAPPING["left_gripper_open"]]
            + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
        )
        uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
        uni_vec[..., UNI_STATE_INDICES] = values
        return uni_vec


if __name__ == "__main__":
    ds = HDF5A2DDataset(use_tasks=["1"])
    # for i in range(len(ds)):
    #     print(f"Processing episode {i}/{len(ds)}...")
    #     item = ds.get_item(task_name="1", index=i)
    #     print(item["meta"])
    #     for k, v in item.items():
    #         if type(v) == np.ndarray:
    #             print(k, v.shape)
