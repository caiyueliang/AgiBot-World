"""
This file will compute the min, max, mean, and standard deviation of each datasets
in `pretrain_datasets.json` or `pretrain_datasets.json`.
"""

import json
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from tqdm import tqdm

from data.hdf5_awsim_dataset import HDF5A2DDataset


def process_hdf5_dataset(vla_dataset):
    EPS = 1e-8
    episode_cnt = 0
    state_sum = 0
    state_sum_sq = 0
    z_state_sum = 0
    z_state_sum_sq = 0
    state_cnt = 0
    nz_state_cnt = None
    state_max = None
    state_min = None
    action_max = None
    action_min = None
    # for i in tqdm(range(len(vla_dataset))):
    #     episode = vla_dataset.get_item(index=i, state_only=True)

    progress_bar = tqdm(total=len(vla_dataset) // 1000)
    
    for task in vla_dataset.file_paths.keys():
        # print(vla_dataset.file_paths_indices[task])
        # print(len(vla_dataset.file_paths[task]))
        # continue
    
        for i in vla_dataset.file_paths_indices[task]:
            episode = vla_dataset.get_item(task_name=task, index=i, state_only=True)
            episode_cnt += 1

            states = episode["state"]

            # Zero the values that are close to zero
            z_states = states.copy()
            z_states[np.abs(states) <= EPS] = 0

            if (abs(z_states) > 5).any():
                print(1)

            # Compute the non-zero count
            if nz_state_cnt is None:
                nz_state_cnt = np.zeros(states.shape[1])
            nz_state_cnt += np.sum(np.abs(states) > EPS, axis=0)

            # Update statistics
            state_sum += np.sum(states, axis=0)
            state_sum_sq += np.sum(states**2, axis=0)
            z_state_sum += np.sum(z_states, axis=0)
            z_state_sum_sq += np.sum(z_states**2, axis=0)
            state_cnt += states.shape[0]
            if state_max is None:
                state_max = np.max(states, axis=0)
                state_min = np.min(states, axis=0)
            else:
                state_max = np.maximum(state_max, np.max(states, axis=0))
                state_min = np.minimum(state_min, np.min(states, axis=0))

            if action_max is None:
                action_max = np.max(episode["action"], axis=0)
                action_min = np.min(episode["action"], axis=0)
            else:
                action_max = np.maximum(action_max, np.max(episode["action"], axis=0))
                action_min = np.minimum(action_min, np.min(episode["action"], axis=0))

            progress_bar.update(1)

    # Add one to avoid division by zero
    nz_state_cnt = np.maximum(nz_state_cnt, np.ones_like(nz_state_cnt))

    result = {
        "dataset_name": vla_dataset.get_dataset_name(),
        "state_mean": (state_sum / state_cnt).tolist(),
        "state_std": np.sqrt(
            np.maximum(
                (z_state_sum_sq / nz_state_cnt)
                - (z_state_sum / state_cnt) ** 2 * (state_cnt / nz_state_cnt),
                np.zeros_like(state_sum_sq),
            )
        ).tolist(),
        "state_min": state_min.tolist(),
        "state_max": state_max.tolist(),
    }

    print("action_min", action_min)
    print("action_max", action_max)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="configs/dataset_stat.json",
        help="JSON file path to save the dataset statistics.",
    )
    parser.add_argument(
        "--skip_exist",
        action="store_true",
        help="Whether to skip the existing dataset statistics.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="all",
        required=False,
        help="Specify the data recipe for training.",
    )
    args = parser.parse_args()

    vla_dataset = HDF5A2DDataset(use_tasks=args.recipe)
    dataset_name = vla_dataset.get_dataset_name()

    try:
        with open(args.save_path, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}
    if args.skip_exist and dataset_name in results:
        print(f"Skipping existed {dataset_name} dataset statistics")
    else:
        print(f"Processing {dataset_name} dataset")
        result = process_hdf5_dataset(vla_dataset)
        results[result["dataset_name"]] = result
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=4)
    print("All datasets have been processed.")
