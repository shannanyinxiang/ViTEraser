output_dir=./output/viteraser-small-segmim/
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
        --lr 0.0001 \
        --warmup_min_lr 0.0000005 \
        --num_workers 6 \
        --epochs 100 \
        --warmup_epochs 1 \
        --print_freq 20 \
        --clip_max_norm 5.0 \
        --encoder swinv2 \
        --decoder swinv2 \
        --pred_mask false \
        --intermediate_erase false \
        --swin_enc_embed_dim 96 \
        --swin_enc_depths 2 2 18 2 \
        --swin_enc_num_heads 3 6 12 24 \
        --swin_enc_window_size 8 \
        --swin_enc_drop_path_rate 0.2 \
        --swin_enc_pretrained_ws 8 \
        --swin_dec_depths 2 18 2 2 2 \
        --swin_dec_num_heads 24 12 6 3 2 \
        --swin_dec_window_size 8 \
        --swin_dec_drop_path_rate 0.2 \
        --swin_dec_pretrained_ws 0 \
        --crop_prob 1.0 \
        --crop_min_ratio 0.7 \
        --crop_max_ratio 1.0 \
        --horizontal_flip_prob 0.3 \
        --rotate_prob 0.0 \
        --pretrained_encoder pretrained/swinv2_small_patch4_window8_256.pth \
        --output_dir ${output_dir} | tee -a ${log_path}