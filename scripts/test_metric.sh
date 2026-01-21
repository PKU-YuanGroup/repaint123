export CUDA_VISIBLE_DEVICES=2

exp_name=logs_alpha_ours_rmedge_soft3_1loss_ip2_masa_resize
dataset=alpha

# python test_generate_images.py --dir datasets/${dataset} --dataset ${dataset} --out ${exp_name} --gpu 7
python metric_utils.py --input_path datasets --datasets ${dataset} --pred_pattern "results/${exp_name}" --results_folder "."