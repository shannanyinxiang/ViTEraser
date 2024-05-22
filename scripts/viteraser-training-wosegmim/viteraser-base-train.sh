output_dir=output/viteraser-base-train/
log_path=${output_dir}log_train.txt

mkdir 'output'
mkdir ${output_dir}

CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch \
        --master_port=3150 \
        --nproc_per_node 2 \
        main.py \
        --data_root data/TextErase/ \
        --train_dataset scutens_train \
        --batch_size 8 \
        --num_workers 6 \
        --lr 0.0005 \
        --epochs 300 \
        --warmup_epochs 10 \
        --crop_prob 0.0 \
        --horizontal_flip_prob 0.3 \
        --rotate_max_angle 30 \
        --rotate_prob 0.5 \
        --encoder swinv2 \
        --decoder swinv2 \
        --swin_enc_embed_dim 128 \
        --swin_enc_depths 2 2 18 2 \
        --swin_enc_num_heads 4 8 16 32 \
        --swin_enc_window_size 8 \
        --swin_enc_drop_path_rate 0.5 \
        --swin_enc_pretrained_ws 8 \
        --swin_enc_use_checkpoint \
        --swin_dec_depths 2 18 2 2 2 \
        --swin_dec_num_heads 32 16 8 4 2 \
        --swin_dec_window_size 8 \
        --swin_dec_drop_path_rate 0.5 \
        --swin_dec_pretrained_ws 0 \
        --pretrained_vgg16 pretrained/vgg16-397923af.pth \
        --pretrained_encoder pretrained/swinv2_base_patch4_window8_256.pth \
        --output_dir ${output_dir} | tee -a ${log_path}