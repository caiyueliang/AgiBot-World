# RDT Baseline

## :video_game: Setup <a name="installation"></a>

1. (Optional) We use conda to manage the environment.

```bash
conda create -n rdt python=3.10 -y
conda activate rdt
```

2. Install dependencies.

```bash
# Clone our repo and pip install to download dependencies
git clone -b manipulation-challenge https://github.com/OpenDriveLab/AgiBot-World.git
cd RDT
pip install -r requirements.txt

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## :fire: Running

### :one: Checkpoints Downloading
- Download the weight of text encoders from <td><a href="https://huggingface.co/google/t5-v1_1-xxl/tree/main">t5-v1_1-xxl</a></td>.

- Download the weight of visual encoders from <td><a href="https://huggingface.co/google/siglip-so400m-patch14-384">siglip-so400m-patch14-384</a></td>.

- Download the weight of pretrained <td><a href="https://huggingface.co/OpenDriveLab-org/rdt-iros-manipulation-challenge-baseline/tree/main/awb-pretrained">RDT</a></td> (pretrain on AgiBot World).

### :two: Precomputing Data Statistics
```bash
python RDT/data/compute_dataset_stat_hdf5_a2d.py
```

### :three: Precomputing Instruction Embedding
```bash
python RDT/scripts/encode_lang.py
```

### :four: Training

```bash
# Start training with 8 GPUs
torchrun \
--standalone \
--nnodes 1 \
--nproc-per-node 8 \    
main.py \
--deepspeed="./configs/zero2.json" \
--pretrained_model_name_or_path="rdt-pretrain-awb" \
--pretrained_text_encoder_name_or_path="google/t5-v1_1-xxl" \
--pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384" \
--output_dir="output" \
--train_batch_size=32 \
--sample_batch_size=64 \
--max_train_steps=100000 \
--checkpointing_period=1000 \
--sample_period=500 \
--checkpoints_total_limit=4 \
--lr_scheduler="constant" \
--learning_rate=1e-4 \
--mixed_precision="bf16" \
--dataloader_num_workers=8 \
--image_aug \
--dataset_type="finetune" \
--state_noise_snr=40 \
--report_to="all" \
--load_from_hdf5 \
--gradient_accumulation_steps=1 \
```

Once you finished training and get the action decoder and VLA backbone, you can simply start the evaluation with:

## :chart_with_upwards_trend: Evaluation
```bash
omni_python scripts/infer.py --task_name test_task_name
```
> In the inference process, we use ROS2 to achieve data communication between the model and the <td><a href="https://github.com/AgibotTech/genie_sim">Genie Sim Benchmark</a></td> simulation environment.
