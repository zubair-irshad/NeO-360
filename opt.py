import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego",
        help="root directory of dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="blender",
        choices=[
            "pd_multi_obj",
            "pd_multi_obj_ae"
        ],
        help="which dataset to train/val",
    )
    parser.add_argument(
        "--save_path", type=str, default="vanilla", help="save results during eval"
    )
    parser.add_argument(
        "--img_wh",
        nargs="+",
        type=int,
        default=[640, 480],
        help="resolution (img_w, img_h) of the image",
    )
    parser.add_argument(
        "--white_back",
        default=False,
        action="store_true",
        help="try for synthetic scenes like blender",
    )
    parser.add_argument(
        "--spheric_poses",
        default=True,
        action="store_true",
        help="whether images are taken in spheric poses (for llff)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=2458,
        help="Total number of different objects in a category",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="dim of latent each for shape and appearance",
    )
    parser.add_argument(
        "--N_emb_xyz",
        type=int,
        default=10,
        help="number of frequencies in xyz positional encoding",
    )
    parser.add_argument(
        "--N_emb_dir",
        type=int,
        default=4,
        help="number of frequencies in dir positional encoding",
    )
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples"
    )
    parser.add_argument(
        "--N_importance", type=int, default=64, help="number of additional fine samples"
    )
    parser.add_argument(
        "--use_disp",
        default=False,
        action="store_true",
        help="use disparity depth sampling",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="factor to perturb depth sampling points",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=1.0,
        help="std dev of noise added to regularize sigma",
    )

    parser.add_argument(
        "--crop_img",
        default=False,
        action="store_true",
        help="initially crop the image or not",
    )
    parser.add_argument(
        "--use_image_encoder",
        default=False,
        action="store_true",
        help="initially crop the image or not",
    )
    parser.add_argument(
        "--latent_code_path", type=str, default=None, help="which category to use"
    )
    parser.add_argument(
        "--encoder_type", type=str, default="resnet", help="which category to use"
    )
    parser.add_argument(
        "--finetune_lpips",
        default=False,
        action="store_true",
        help="whether to finetune with lpips loss and patched dataloader",
    )

    # params for SRN multicat training

    parser.add_argument(
        "--splits", type=str, default=None, help="which category to use"
    )

    # parser.add_argument("--run_eval", default=False, action="store_true")
    parser.add_argument("--eval_mode", default=None, type=str)
    # options "full_eval", "vis_only"

    parser.add_argument("--do_generate", default=False, action="store_true")
    parser.add_argument(
        "--val_splits", type=str, default=None, help="which category to use"
    )
    parser.add_argument("--cat", type=str, default=None, help="which category to use")
    parser.add_argument("--use_tcnn", default=False, action="store_true")

    parser.add_argument(
        "--model_type",
        type=str,
        default="geometry",
        help="which model to use i.e. geometry or render for refnerf",
    )
    parser.add_argument(
        "--train_opacity_rgb",
        default=False,
        action="store_true",
        help="whether to train both opacity and rgb for voxel model",
    )

    # params for latent codes:
    #
    parser.add_argument(
        "--N_max_objs",
        type=int,
        default=151,
        help="maximum number of object instances in the dataset",
    )

    # onl for nerfmvs
    parser.add_argument(
        "--nv",
        type=int,
        default=3,
        help="maximum number of object instances in the dataset",
    )

    parser.add_argument(
        "--num_nocs_ch",
        type=int,
        default=256,
        help="maximum number of object instances in the dataset",
    )
    parser.add_argument(
        "--N_obj_code_length", type=int, default=128, help="size of latent vector"
    )
    ## params for Nerf Model
    # (Scene branch)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--N_freq_xyz", type=int, default=10)
    parser.add_argument("--N_freq_dir", type=int, default=4)
    parser.add_argument("--skips", type=list, default=[4])

    ## params for Nerf Model
    # (Obj branch)
    parser.add_argument("--inst_D", type=int, default=4)
    parser.add_argument("--inst_W", type=int, default=128)
    parser.add_argument("--inst_skips", type=list, default=[2])
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    # parser.add_argument(
    #     "--chunk",
    #     type=int,
    #     default=16 * 128,
    #     help="chunk size to split the input to avoid OOM",
    # )
    parser.add_argument(
        "--chunk",
        type=int,
        default=16 * 64,
        help="chunk size to split the input to avoid OOM",
    )
    # parser.add_argument('--chunk', type=int, default= 32*1024,
    #                     help='chunk size to split the input to avoid OOM')
    parser.add_argument(
        "--num_epochs", type=int, default=80, help="number of training epochs"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus")

    parser.add_argument(
        "--run_max_steps", type=int, default=100000, help="number of gpus"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint to load (including optimizers, etc)",
    )
    parser.add_argument(
        "--is_optimize",
        type=str,
        default=None,
        help="whether to finetune the network after training on prior data",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="pretrained checkpoint to load (including optimizers, etc)",
    )
    parser.add_argument(
        "--prefixes_to_ignore",
        nargs="+",
        type=str,
        default=["loss"],
        help="the prefixes to ignore in the checkpoint state dict",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="pretrained model weight to load (do not load optimizers, etc)",
    )

    #### Loss params
    parser.add_argument("--color_loss_weight", type=float, default=1.0)
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)
    parser.add_argument("--opacity_loss_weight", type=float, default=10.0)
    parser.add_argument("--instance_color_loss_weight", type=float, default=1.0)
    parser.add_argument("--instance_depth_loss_weight", type=float, default=1.0)

    #### object-nerf optimizer params
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer type",
        choices=["sgd", "adam", "radam", "ranger"],
    )
    # parser.add_argument('--lr', type=float, default=1.0e-3,
    #                     help='learning rate')
    parser.add_argument("--lr", type=float, default=1.0e-3, help="learning rate")
    parser.add_argument("--iters", type=int, default=30000, help="iters")
    # parser.add_argument('--lr', type=float, default=1.0e-4,
    #                     help='learning rate')
    parser.add_argument("--latent_lr", type=float, default=1.0e-3, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="learning rate momentum"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="poly",
        help="scheduler type",
        choices=["steplr", "cosine", "poly"],
    )
    parser.add_argument(
        "--lr_scheduler_latent",
        type=str,
        default="poly",
        help="scheduler type",
        choices=["steplr", "cosine", "poly"],
    )
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument(
        "--warmup_multiplier",
        type=float,
        default=1.0,
        help="lr is multiplied by this factor after --warmup_epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Gradually warm-up(increasing) learning rate in optimizer",
    )

    #### nerf_pl configs
    # parser.add_argument('--optimizer', type=str, default='adam',
    #                     help='optimizer type',
    #                     choices=['sgd', 'adam', 'radam', 'ranger'])
    # parser.add_argument('--lr', type=float, default=5e-4,
    #                     help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='learning rate momentum')
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                     help='weight decay')
    # parser.add_argument('--lr_scheduler', type=str, default='steplr',
    #                     help='scheduler type',
    #                     choices=['steplr', 'cosine', 'poly'])
    # #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    # parser.add_argument('--warmup_multiplier', type=float, default=1.0,
    #                     help='lr is multiplied by this factor after --warmup_epochs')
    # parser.add_argument('--warmup_epochs', type=int, default=0,
    #                     help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument(
        "--decay_step", nargs="+", type=int, default=[20], help="scheduler decay step"
    )
    parser.add_argument(
        "--decay_gamma", type=float, default=0.1, help="learning rate decay amount"
    )
    ###########################
    #### params for poly ####
    parser.add_argument(
        "--poly_exp",
        type=float,
        default=0.99,
        help="exponent for polynomial learning rate decay",
    )
    # parser.add_argument('--poly_exp', type=float, default=2,
    #                     help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")

    parser.add_argument(
        "--render_name", type=str, default=None, help="render directory name"
    )

    parser.add_argument(
        "--exp_type",
        type=str,
        default="vanilla",
        help="experiment type --choose from vanilla, pixel_nerf, pixel_nerf_sphere, groundplanar, triplanar",
    )

    ###########################

    # parser.add_argument('--ckpt_path', type=str, default='last.ckpt',
    #                     help='ckpt path')

    return parser.parse_args()
