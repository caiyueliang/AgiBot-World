python main.py test \
    --ckpt_path /cpfs01/user/buqingwen/OmniEmbodiment/logs/stage2_ckpts_openx_v2/epoch=1-step=270000.ckpt \
    --config config/latent_action_openx_stage-2.yaml \
    2>&1 | tee output_test.log