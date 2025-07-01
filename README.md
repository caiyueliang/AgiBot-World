# Manipulation Baseline
We adopt [UniVLA](https://github.com/OpenDriveLab/AgiBot-World/tree/manipulation-challenge/UniVLA) and [RDT](https://github.com/OpenDriveLab/AgiBot-World/tree/manipulation-challenge/RDT) as baseline models for the [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - Manipulation track.

## :trophy: Leaderboard

Results of baseline models. More detailed task descriptions and metric definitions can be found [here](https://agibot-world.com/challenge/manipulation/leaderboard).

<table border="0">
  <tr>
    <th>Model Name</th>
    <th>Total Score</th>
    <th>Clear the countertop waste</th>
    <th>Open drawer and store items</th>
    <th>Heat the food in the microwave</th>
    <th>Pack moving objects from conveyor</th>
    <th>Pickup items from the freezer</th>
    <th>Restock supermarket items</th>
    <th>Pack in the supermarket</th>
    <th>Make a sandwich</th>
    <th>Clear table in the restaurant</th>
    <th>Stamp the seal</th>
  </tr>
  <tr>
    <td>UniVLA</td>
    <td>2.336</td>
    <td>0.194</td>
    <td>0</td>
    <td>0.198</td>
    <td>0</td>
    <td>0.08</td>
    <td>0.55</td>
    <td>1</td>
    <td>0.064</td>
    <td>0.25</td>
    <td>0</td>
  </tr>
  <tr>
    <td>RDT</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>

## 🤗 Model Card

<table>
  <tr>
    <th>Model Name</th>
    <th>Backbone</th>
    <th>HF Path</th>
    <th>Note</th>
  </tr>

  <tr>
    <td>univla-iros-manipulation-challenge-baseline</td>
    <td><a href="https://huggingface.co/qwbu/univla-7b">UniVLA-7b</a></td>
    <td><a href="https://huggingface.co/qwbu/univla-iros-manipulation-challenge-baseline">univla-iros-manipulation-challenge-baseline </a></td>
    <td> Without pretraining on AgibotWorld dataset. Finetuned collectively on all challenge tasks. </td>
  </tr>

  <tr>
    <td>rdt-iros-manipulation-challenge-baseline</td>
    <td>TBD</td>
    <td>TBD</td>
    <td> Pretrained on AgibotWorld dataset. Finetuned collectively on all challenge tasks. </td>
  </tr>

</table>

## :file_folder: Dataset

### :one: Dataset Downloading

- Download the simdata from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-SimData">Manipulation-SimData</a></td> for challenge phase1.

- Download the realrobot data from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-RealRobot">Manipulation-RealRobotData</a></td> for challenge phase2.

- Pretraining on more public data is allowed. If needed, download the AgibotWorld-Alpha dataset from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha">AgibotWorld-Alpha</a></td>, or the AgibotWorld-Beta dataset (larger) from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta">AgibotWorld-Beta</a></td>.

### :two: Dataset Directory Structure
The dataset directory structure is organized as follows:

```
dataset
├── 2810051
│   ├── 3026521
│   │   ├── A2D0015AB00061
│   │   │   ├── 12030289
│   │   │   │   ├── camera
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── hand_left_color.jpg
│   │   │   │   │   │   ├── hand_right_color.jpg
│   │   │   │   │   │   ├── head_color.jpg
│   │   │   │   │   │   └── ...
│   │   │   │   │   └── ...
│   │   │   │   ├── aligned_joints.h5
│   │   │   │   ├── data_info.json
│   │   │   │   ├── meta_info.json
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── 2810052
├── ...
├── task_0_train.json
├── task_1_train.json
├── ...
├── task_9_train.json
```
Subfolder such as `2810051`, `2810083` comes from different tasks. You can move all of them into folder `dataset` as above, and then choose which task to use by modify `--task_ids` as below.

## :pushpin: TODO list
-  [x] Training code and dataloader for challenge dataset.
-  [x] Evaluation code.
-  [x] Finetuned UniVLA checkpoints on challenge simdata.
-  [x] Updated simulation environment.
-  [ ] Finetuned RDT checkpoints on challenge simdata.
