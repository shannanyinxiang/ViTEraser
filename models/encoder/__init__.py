from .swinv2_encoder import SwinV2Encoder

def build_encoder(args):
    train_encoder = args.lr_encoder_ratio > 0
    if args.encoder == 'swinv2':
        if args.dataset_file == 'texterase' or args.segmim_finetune:
            encoder_type = 'SwinTransformerV2'
        else:
            encoder_type = 'SwinTransformerV2ForSimMIM'
            
        encoder = SwinV2Encoder(train_encoder, args.pretrained_encoder, args.swin_enc_embed_dim, args.swin_enc_depths, 
            args.swin_enc_num_heads, args.swin_enc_drop_path_rate, args.swin_enc_pretrained_ws, args.swin_enc_window_size,
            args.swin_use_checkpoint or args.swin_enc_use_checkpoint, encoder_type)
    else:
        raise NotImplementedError
    return encoder