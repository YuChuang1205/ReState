# build Qwen3-VL solution cache
CUDA_VISIBLE_DEVICES=0 python build_solution_cache_qwenvl.py \
   --data_root ../data/VersaCoT10k/data \
   --caption_file ../data/instruct_captions_VersaCoT10k_qwen72b_ok.json \
   --split train \
   --train_split train \
   --val_split val \
   --test_split test \
   --solution_gen_model Qwen/Qwen3-VL-8B-Instruct \
   --solution_cache_dir ./solution_aug_cache_train_qwenvl_001 \
   --solution_pos_num 2 \
   --solution_neg_num 2 \
   --solution_gen_max_new_tokens 1024 \
   --solution_gen_temperature 0.9 \
   --solution_gen_top_p 0.9 \
   --solution_gen_do_sample \
   --solution_gen_use_image \
   --solution_gen_use_caption \
   --solution_gen_device cuda:1 \
   --use_caption
