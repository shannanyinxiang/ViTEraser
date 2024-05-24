output_dir=./output/viteraser-tiny-segmim-encoder-finetune/
log_path=${output_dir}log_train.txt

mkdir 'output'
mkdir ${output_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u -m torch.distributed.launch \
        --master_port=3140 \
        --nproc_per_node=4 \
        main_segmim.py \
        --dataset_file segmim \
        --train_dataset art_train ic13_train ic15_train lsvt_train mlt2017_train rects_train textocr_train textocr_val \
        --data_root data/SegMIMDatasets \
        --batch_size 16 \
        --lr 0.00125 \
        --warmup_min_lr 0.00000025 \
        --min_lr 0.00000025 \
        --layer_decay 0.75 \
        --num_workers 6 \
        --epochs 20 \
        --warmup_epochs 4 \
        --save_interval 2 \
        --print_freq 20 \
        --clip_max_norm 5.0 \
        --encoder swinv2 \
        --decoder swinv2 \
        --pred_mask false \
        --intermediate_erase false \
        --swin_enc_embed_dim 96 \
        --swin_enc_depths 2 2 6 2 \
        --swin_enc_num_heads 3 6 12 24 \
        --swin_enc_window_size 8 \
        --swin_enc_drop_path_rate 0.1 \
        --swin_enc_pretrained_ws 8 \
        --crop_prob 1.0 \
        --crop_min_ratio 0.7 \
        --crop_max_ratio 1.0 \
        --horizontal_flip_prob 0.3 \
        --rotate_prob 0.5 \
        --rotate_max_angle 30 \
        --segmim_ft_init_weight_path output/viteraser-tiny-segmim/checkpoints/checkpoint0099.pth \
        --segmim_finetune true \
        --output_dir ${output_dir} | tee -a ${log_path}