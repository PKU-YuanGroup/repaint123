

gpu=5
dataset=alpha
angle=40
ddnm_strength=0.6
exp_name=${dataset}_ours_rmedge_soft3_1loss_ip2_masa_resize_inv0.6_ip0.5_angle${angle}_viewprompt_cache_${ddnm_strength}_scale4
# cp -r logs_realfusion_ours_rmedge_soft3_1loss_ip2_masa logs_${exp_name}
# cp -r logs_alpha_ours_rmedge_soft3_1loss_ip2_masa_resize logs_${exp_name}

fine_config_name=image2_angle${angle}
python scripts/runall.py --ddnm_strength ${ddnm_strength} --fine_config_name ${fine_config_name} --dir datasets/${dataset} --out logs_${exp_name} --video-out videos_${exp_name} --gpu ${gpu}
python test_generate_images.py --dir datasets/${dataset} --dataset ${dataset} --out logs_${exp_name} --gpu ${gpu}
export https_proxy=127.0.0.1:7900
export http_proxy=127.0.0.1:7900
python metric_utils.py --device ${gpu} --input_path datasets --datasets ${dataset} --pred_pattern "results/logs_${exp_name}" --results_folder "."