# Manipulation Baseline
We adopt [UniVLA](https://github.com/OpenDriveLab/UniVLA) as the baseline model for the [AgiBot World Challenge @ IROS 2025](https://agibot-world.com/challenge) - Manipulation track.

This repo provides a minimal version of training code.

## :video_game: Setup <a name="installation"></a>

1. (Optional) We use conda to manage the environment.

```bash
conda create -n univla python=3.10 -y
conda activate univla
```

2. Install dependencies.

```bash
# Clone our repo and pip install to download dependencies
git clone -b manipulation-challenge https://github.com/OpenDriveLab/AgiBot-World.git
pip install -r requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## :fire: Training 

### :one: Download the dataset

- Download the simdata from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-SimData">Manipulation-SimData</a></td> for challenge phase1.

- Download the realrobot data from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2025/tree/main/Manipulation-RealRobot">Manipulation-RealRobotData</a></td> for challenge phase2.

- Pretraining on more public data is allowed. If needed, download the AgibotWorld-Alpha dataset from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha">AgibotWorld-Alpha</a></td>, or the AgibotWorld-Beta dataset (larger) from <td><a href="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta">AgibotWorld-Beta</a></td>.

### :two: Download the checkpoints
- Download the checkpoint of latent action model from <td><a href="https://huggingface.co/qwbu/univla-latent-action-model">univla-latent-action-model</a></td>.

- Download the weight of <td><a href="https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b">TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b</a></td>.

- Download the univla-7b checkpoint from <td><a href="https://huggingface.co/qwbu/univla-7b">univla-7b</a></td>.

### :three: Dataset Directory Structure
The dataset directory structure is organized as follows:

# Dataset Directory Structure

You can organize dataset directory structure as follows:
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
├── task_1_train.json
├── task_2_train.json
├── ...
├── task_10_train.json
```
Subfolder such as `2810051`, `2810083` comes from different tasks. You can move all of them into folder `dataset` as above, and then choose which task to use by modify `--task_ids` as below.

### :four: Running

```bash
# Start training with 8 GPUs
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 8 \
scripts/finetune_genie.py \
--vla_path univla-7b \
--lam_path univla-latent-action-model \
--data_root_dir dataset \
--meta_json_dir dataset \
--codebook_size 16 \
--batch_size 4 \
--grad_accumulation_steps 1 \
--max_steps 10000 \
--save_steps 1000 \
--decoder_n_layers 2 \
--decoder_hidden_dim 1024 \
--run_root_dir checkpoints/rundir \
--adapter_tmp_dir checkpoints/adapterdir \
--save_latest_checkpoint_only \
--with_proprio \
--use_lora \
--task_ids 1 \ # for task 1
# --task_ids 0 1 2 3 4 5 6 7 8 9 \ # for all 10 tasks
```

Once you finished training and get the action decoder and VLA backbone, you can simply start the evaluation with:

## Evaluation
```bash
omni_python scripts/infer.py
```
> In the inference process, we use ROS2 to achieve data communication between the model and the <td><a href="https://github.com/AgibotTech/genie_sim">Genie Sim Benchmark</a></td> simulation environment. The interface is to be updated.

## :pushpin: TODO list
-  [x] Training code and dataloader for challenge dataset.
-  [x] Evaluation code.
-  [ ] Finetuned UniVLA checkpoints on challenge simdata.
-  [ ] Updated simulation environment.
-  [ ] Finetuned RDT checkpoints on challenge simdata.

## :pencil: Citation
If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/pdf/2505.06111):

```bibtex
@article{bu2025univla,
  title={UniVLA: Learning to Act Anywhere with Task-centric Latent Actions}, 
  author={Qingwen Bu and Yanting Yang and Jisong Cai and Shenyuan Gao and Guanghui Ren and Maoqing Yao and Ping Luo and Hongyang Li},
  journal={arXiv preprint arXiv:2505.06111},
  year={2025}
}
```
