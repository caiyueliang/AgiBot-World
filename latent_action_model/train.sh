torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
    --config config/latent_action_bridge.yaml\
    2>&1 | tee ego4d_test.log