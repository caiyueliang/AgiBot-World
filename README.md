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
    <td>2.795</td>
    <td>0.097</td>
    <td>0.02</td>
    <td>0.033</td>
    <td>0.35</td>
    <td>0.26</td>
    <td>0.4</td>
    <td>1</td>
    <td>0.08</td>
    <td>0.375</td>
    <td>0.18</td>
  </tr>
  <tr>
    <td>RDT</td>
    <td>2.434</td>
    <td>0.296</td>
    <td>0</td>
    <td>0.133</td>
    <td>0</td>
    <td>0.48</td>
    <td>0.25</td>
    <td>0.825</td>
    <td>0</td>
    <td>0.25</td>
    <td>0.2</td>
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
    <td><a href="https://huggingface.co/qwbu/univla-iros-manipulation-challenge-baseline">univla-iros-manipulation-challenge-baseline</a></td>
    <td> Without pretraining on AgibotWorld dataset. Finetuned collectively on all challenge tasks. </td>
  </tr>

  <tr>
    <td>rdt-iros-manipulation-challenge-baseline</td>
    <td><a href="https://huggingface.co/OpenDriveLab-org/rdt-iros-manipulation-challenge-baseline/tree/main/awb-pretrained">RDT</a></td>
    <td><a href="https://huggingface.co/OpenDriveLab-org/rdt-iros-manipulation-challenge-baseline/tree/main/sim-finetuned">rdt-iros-manipulation-challenge-baseline</a></td>
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

## :rocket: Submission
After completing local debugging, you can submit your work on our <td><a href="https://agibot-world.com/challenge/manipulation/my-submissions">test server</a></td>. Please note the following points:

- You need to compress your code repository into a ZIP file before submission.
- In your code repository, there must be a directory named `scripts`, and within this directory, there must be a file named `infer.py` for model inference. For the `infer.py` file, you need to have a parameter named `task_name` which can take the values of the 10 task names in our competition. For specific details, please refer to the <td><a href="https://github.com/OpenDriveLab/AgiBot-World/blob/manipulation-challenge/UniVLA/scripts/infer.py">inference script example</a></td> we provided. After you complete the submission, our test server will execute the default command `omni_python scripts/infer.py --task_name [task_name]` and iterate through all the tasks in sequence.
- In your code repository, you must include the file `genie_sim_ros.py`, which is used for data communication between the model and the simulation environment. You can directly copy the file we provided. Of course, if needed, you can modify it yourself.
- In your code repository, you also need to provide the model checkpoint. By default, you should place the checkpoint in the directory `checkpoints/finetuned` and set the parameters "pretrained_checkpoint" and "action_decoder_path" in `scripts/infer.py` to `checkpoints/finetuned` (taking UniVLA as an example). Of course, you can change the location of the checkpoint as needed, as long as you also update the path parameters in `scripts/infer.py` accordingly.
- If there are any new Python dependencies required, you need to list them in a `requirements.txt` file and upload it when submitting to the test server. For UniVLA and RDT, no new dependencies are needed, so you can simply submit an empty `requirements.txt` file (but it must be submitted). It is recommended that you thoroughly debug and validate locally before uploading.

Illustration of the code repository file structure: 

```
your_repository/
├── scripts/
│   ├── infer.py
│   └── ...
├── checkpoints/
│   └── finetuned/
│       ├── config.json
│       ├── xxx.pt
│       ├── xxx.safetensors
│       └── ...
├── genie_sim_ros.py
└── ...
```
## :pushpin: TODO list
-  [x] Training code and dataloader for challenge dataset.
-  [x] Evaluation code.
-  [x] Finetuned UniVLA checkpoints on challenge simdata.
-  [x] Updated simulation environment.
-  [x] Finetuned RDT checkpoints on challenge simdata.
