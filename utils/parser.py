import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set ViTEraser', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder_ratio', default=0.1, type=float)
    parser.add_argument('--lr_scheduler', default='linear', type=str, choices=['linear', 'cosine'])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--warmup_min_lr', type=float, default=0.0001)
    parser.add_argument('--min_lr', type=float, default=0.00001)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # Model Parameters
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--load_pre_ignore_backbone', action='store_true')
    parser.add_argument('--pretrained_encoder', type=str, default='')
    parser.add_argument('--pretrained_decoder', type=str, default='')
    parser.add_argument('--pretrained_feat_extractor', type=str, default='')
    parser.add_argument('--mask_sigmoid', action='store_true')
    parser.add_argument('--encoder', default='swinv2', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--decoder', type=str, default='swinv2')
    parser.add_argument('--post_skip', action='store_true')
    parser.add_argument('--mask_blk', type=str, default='deconv')
    parser.add_argument('--swin_dec_depths', type=int, nargs='+', default=[2, 6, 2, 2, 2])
    parser.add_argument('--swin_dec_num_heads', type=int, nargs='+', default=[24, 12, 6, 3, 2])
    parser.add_argument('--swin_dec_window_size', type=int, default=7)
    parser.add_argument('--swin_dec_drop_path_rate', type=float, default=0.2)
    parser.add_argument('--swin_dec_pretrained_ws', type=int, default=8)
    parser.add_argument('--dis_sigmoid', type=str2bool, default=True)
    parser.add_argument('--swin_enc_depths', type=int, nargs='+', default=[2, 2, 6, 2])
    parser.add_argument('--swin_enc_num_heads', type=int, nargs='+', default=[3, 6, 12, 24])
    parser.add_argument('--swin_enc_drop_path_rate', type=float, default=0.2)
    parser.add_argument('--swin_enc_embed_dim', type=int, default=96)
    parser.add_argument('--swin_enc_pretrained_ws', type=int, default=8)
    parser.add_argument('--swin_enc_window_size', type=int, default=16)
    parser.add_argument('--pred_mask', type=str2bool, default=True)
    parser.add_argument('--intermediate_erase', type=str2bool, default=True)
    parser.add_argument('--swin_use_checkpoint', action='store_true')
    parser.add_argument('--swin_enc_use_checkpoint', action='store_true')
    parser.add_argument('--swin_dec_use_checkpoint', action='store_true')

    # Data parameters
    parser.add_argument('--crop_min_ratio', type=float, default=0.5)
    parser.add_argument('--crop_max_ratio', type=float, default=1.0)
    parser.add_argument('--crop_prob', type=float, default=1.0)
    parser.add_argument('--rotate_max_angle', type=int, default=30)
    parser.add_argument('--rotate_prob', type=float, default=0.3)
    parser.add_argument('--mask_fillcolor', type=int, default=0)
    parser.add_argument('--brightness', type=float, default=0.5)
    parser.add_argument('--contrast', type=float, default=0.5)
    parser.add_argument('--saturation', type=float, default=0.5)
    parser.add_argument('--hue', type=float, default=0.5)
    parser.add_argument('--distortion_prob', type=float, default=0.5)
    parser.add_argument('--horizontal_flip_prob', type=float, default=0.3)
    parser.add_argument('--pix2pix_size', type=int, default=512)
    parser.add_argument('--dataset_file', default='texterase')
    parser.add_argument('--train_dataset', type=str, nargs='+')
    parser.add_argument('--val_dataset', type=str, nargs='+')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--init_method', type=str, default='')

    return parser

def str2bool(argv):
    return True if argv.lower() == 'true' else False