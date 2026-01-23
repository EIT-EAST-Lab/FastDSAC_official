#!/bin/bash


START_TIME=$(date +%s)
echo "=========================================="
echo "开始训练序列: $(date)"
echo "=========================================="


# example
python fast_sac/train_fastdsac_torch_enhanced.py --env_name h1hand-push-v0 --exp_name FastDSACT_evalhb_hard_torch_alpha0.001_std1_lesswd_nobound_warm_taub0.005tau0.005relu_enhanced_wentropyl_ln --render_interval 50000 --seed 777 --project "Fast HB final paper runs" --total_timesteps 500000 --eval_interval 10000 --learning_starts 1000 --critic_learning_rate 0.0003 --actor_learning_rate 0.0003 --alpha_learning_rate 0.0003 --alpha_init 0.001 --save_interval 200001 --activation_type relu --reward_scale 1.0 --weight_decay 0.0001 --target_entropy_ratio 0.0 --use_layer_norm --scale_max 2 --scale_min 0.01 --tuned_betas --tau_b 0.005 --bound_beta 3 --tau 0.005 --log_std_max 1 --log_std_min -10.0

python fast_sac/train_fastdsac_torch_enhanced.py --env_name h1hand-stair-v0 --exp_name FastDSACT_evalhb_hard_torch_alpha0.001_std1_lesswd_nobound_warm_taub0.005tau0.005relu_enhance_wentropylt0.5_ln --render_interval 50000 --seed 666 --project "Fast HB tuned runs" --total_timesteps 150000 --eval_interval 10000 --learning_starts 1000 --critic_learning_rate 0.0003 --actor_learning_rate 0.0003 --alpha_learning_rate 0.0003 --alpha_init 0.001 --save_interval 200001 --activation_type relu --reward_scale 1.0 --weight_decay 0.0001 --target_entropy_ratio 0.0 --use_layer_norm --scale_max 2 --scale_min 0.01 --tuned_betas --tau_b 0.005 --bound_beta 3 --tau 0.005 --log_std_max 1 --log_std_min -10.0 --temperature 0.5



# h1hand-walk
# h1hand_stand
# h1hand_run
# h1hand_reach
# h1hand_hurdle
# h1hand_crawl
# h1hand_maze
# h1hand_sit_simple
# h1hand_sit_hard
# h1hand_balance_simple
# h1hand_balance_hard
# h1hand_stair
# h1hand_slide
# h1hand_pole
# h1hand_push
# h1hand_cabinet
# h1hand_door
# h1hand_truck
# h1hand_cube
# h1hand_bookshelf_simple
# h1hand_bookshelf_hard
# h1hand_basketball
# h1hand_window
# h1hand_spoon
# h1hand_package
# h1hand_powerlift
# h1hand_room
# h1hand_insert_small
# h1hand_insert_normal
