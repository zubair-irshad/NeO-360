# ------------------------------------------------------------------------------------
# NeO-360
# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# ------------------------------------------------------------------------------------

import os
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import models.neo360.helper as helper
from models.utils import (
    store_image,
    store_depth_img,
    store_depth_raw,
    write_stats,
    get_obj_rgbs_from_segmap,
)
from models.interface import LitModel
from torch.utils.data import DataLoader
from datasets import dataset_dict
from collections import defaultdict
from utils.train_helper import *
from models.neo360.util import *
from models.neo360.encoder_tp_fusion_conv import GridEncoder, index_grid
import wandb

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from dotmap import DotMap
import lpips
from torch_efficient_distloss import eff_distloss

# 2 layer NN with view dirs encoding
class NeRFPPMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        # netdepth: int = 8,
        # netwidth: int = 256,
        netdepth: int = 4,
        netwidth: int = 128,
        netdepth_condition: int = 2,
        netwidth_condition: int = 64,
        skip_layer: int = 2,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        local_latent_size: int = 512,
        world_latent_size: int = 128,
        combine_layer: int = 3,
        combine_type="average",
        out_nocs=False,
        num_src_views=3,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFPPMLP, self).__init__()

        self.num_src_views = num_src_views
        self.net_activation = nn.ReLU(inplace=True)
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch

        pos_size += local_latent_size
        pos_size += world_latent_size
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        if out_nocs:
            self.nocs_layer = nn.Linear(netwidth, num_rgb_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)

        if out_nocs:
            init.xavier_uniform_(self.nocs_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(
        self,
        x,
        condition_tile,
        world_latent,
        local_latent,
        combine_inner_dims,
        out_nocs=False,
    ):
        # world_latent = world_latent.repeat(self.num_src_views, 1, 1,1) # (3, B*N_samples, feats_dim)
        # world_latent = world_latent.reshape(-1,world_latent.shape[-1])
        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)

        x = torch.cat([x, local_latent, world_latent], dim=-1)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)

            if idx == self.combine_layer:
                bottleneck = self.bottleneck_layer(x)
                # print("bottleneck", bottleneck.shape)
                x = combine_interleaved(x, combine_inner_dims, self.combine_type)

            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )
        if out_nocs:
            raw_nocs = self.nocs_layer(x).reshape(
                -1, num_samples, self.num_density_channels
            )

        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            if idx == 0:
                x = combine_interleaved(x, combine_inner_dims, self.combine_type)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        if out_nocs:
            return raw_rgb, raw_density, raw_nocs
        else:
            return raw_rgb, raw_density


# @gin.configurable()
class NeRF_TP(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 128,
        # num_fine_samples: int = 128,
        num_fine_samples: int = 256,
        use_viewdirs: bool = True,
        num_src_views: int = 3,
        density_noise: float = 0.0,
        lindisp: bool = False,
        xyz_min=None,
        xyz_max=None,
        is_optimize=False,
        encoder_type="resnet",
        feats_c_size=0,
        attn=False,
        input_ch_view=3,
        use_same_stride=False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRF_TP, self).__init__()

        self.is_optimize = is_optimize
        self.encoder = GridEncoder(encoder_type=encoder_type)
        self.encoder_type = encoder_type
        self.rgb_activation = nn.Sigmoid()
        self.nocs_activation = nn.Sigmoid()
        self.sigma_activation = nn.Softplus()

        latent_size = self.encoder.spatial_encoder.latent_size
        # print("============================================\n\n\n")
        # print("latent size to check resnet or unet", latent_size)
        # print("============================================\n\n\n")

        self.feats_c_size = feats_c_size
        self.attn = attn
        self.input_ch_view = input_ch_view
        self.use_same_stride = use_same_stride

        self.num_src_views = num_src_views
        print("============================================\n\n\n")
        print("self.num_src_views", self.num_src_views)
        print("======================================\n\n\n")

        # self.obj_coarse_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view, num_src_views = num_src_views, out_nocs = True)
        # self.obj_fine_mlp = NeRFPPMLP(min_deg_point, max_deg_point, deg_view,  num_src_views = num_src_views,out_nocs = True)
        self.fg_coarse_mlp = NeRFPPMLP(
            min_deg_point, max_deg_point, deg_view, num_src_views=num_src_views
        )
        self.fg_fine_mlp = NeRFPPMLP(
            min_deg_point,
            max_deg_point,
            deg_view,
            num_src_views=num_src_views,
        )
        self.bg_coarse_mlp = NeRFPPMLP(
            min_deg_point,
            max_deg_point,
            deg_view,
            num_src_views=num_src_views,
            input_ch=4,
        )
        self.bg_fine_mlp = NeRFPPMLP(
            min_deg_point,
            max_deg_point,
            deg_view,
            num_src_views=num_src_views,
            input_ch=4,
        )

    def get_local_feats(self, samples, poses, all_focal, all_c, src_views_num):
        samples = samples.reshape(-1, 3).unsqueeze(0)
        samples_cam = world2camera(samples, poses, src_views_num)
        focal = all_focal[0].unsqueeze(-1).repeat((1, 2))
        focal[..., 1] *= -1.0
        c = all_c[0].unsqueeze(0)

        # Try reducing focal by 2 to match the feats size

        # focal = focal/2
        # c = c/2
        # image_shape = self.image_shape/2
        image_shape = self.image_shape
        # mask_z = samples_cam[:,:,2]<1e-3
        uv = projection(samples_cam, focal, c, src_views_num)
        latent, _ = self.encoder.spatial_encoder.index(
            uv, None, image_shape
        )  # (SB * NS, latent, B)
        # mask = (mask.sum(dim=-1) == 2) & (mask_z)
        # latent[mask.unsqueeze(1).repeat(1, latent.shape[1], 1) == False] = 0
        # in_mask = mask.float().reshape(-1,1)
        latent = latent.transpose(1, 2).reshape(
            -1, self.latent_size
        )  # (SB * NS * B, latent)
        # latent = torch.cat((latent,in_mask), dim=-1)
        return latent, samples_cam

    def forward(self, rays, randomized, white_bkgd, near, far, out_depth=False):
        self.image_shape = torch.Tensor(
            [rays["src_imgs"].shape[-1], rays["src_imgs"].shape[-2]]
        ).to(rays["src_imgs"].device)

        self.latent_size = self.encoder.latent_size
        scene_grid_xz, scene_grid_xy, scene_grid_yz = self.encoder(
            rays["src_imgs"], rays["src_poses"], rays["src_focal"], rays["src_c"]
        )

        ret = []
        near = torch.full_like(rays["rays_o"][..., -1:], 1e-4)
        far = helper.intersect_sphere(rays["rays_o"], rays["rays_d"])

        for i_level in range(self.num_levels):
            if i_level == 0:
                fg_t_vals, fg_samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    in_sphere=True,
                )
                bg_t_vals, bg_samples, bg_samples_linear = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    in_sphere=False,
                    far_uncontracted=3,
                )
                fg_mlp = self.fg_coarse_mlp
                bg_mlp = self.bg_coarse_mlp
                # obj_mlp = self.obj_coarse_mlp

            else:
                fg_t_mids = 0.5 * (fg_t_vals[..., 1:] + fg_t_vals[..., :-1])
                fg_t_vals, fg_samples = helper.sample_pdf(
                    bins=fg_t_mids,
                    weights=fg_weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=fg_t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                    in_sphere=True,
                )
                bg_t_mids = 0.5 * (bg_t_vals[..., 1:] + bg_t_vals[..., :-1])
                bg_t_vals, bg_samples, bg_samples_linear = helper.sample_pdf(
                    bins=bg_t_mids,
                    weights=bg_weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=bg_t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                    in_sphere=False,
                    far=far,
                    far_uncontracted=3,
                )

                fg_mlp = self.fg_fine_mlp
                bg_mlp = self.bg_fine_mlp
                # obj_mlp = self.obj_fine_mlp

            # viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)

            viewdirs = world2camera_viewdirs(
                rays["viewdirs"].unsqueeze(0), rays["src_poses"], self.num_src_views
            )

            def predict(
                samples,
                mlp,
                world_latent,
                local_latent,
                B_fg,
                N_samples,
                out_nocs=False,
            ):
                samples_enc = helper.pos_enc(
                    samples,
                    self.min_deg_point,
                    self.max_deg_point,
                )
                viewdirs_enc = helper.pos_enc(viewdirs, 0, self.deg_view)
                viewdirs_enc = torch.tile(
                    viewdirs_enc[:, None, :], (1, N_samples, 1)
                ).reshape(-1, viewdirs_enc.shape[-1])
                B, N_points, _ = samples_enc.shape
                if out_nocs:
                    raw_rgb, raw_sigma, raw_nocs = mlp(
                        samples_enc,
                        viewdirs_enc,
                        world_latent,
                        local_latent,
                        combine_inner_dims=(self.num_src_views, N_points),
                        out_nocs=True,
                    )
                else:
                    raw_rgb, raw_sigma = mlp(
                        samples_enc,
                        viewdirs_enc,
                        world_latent,
                        local_latent,
                        combine_inner_dims=(self.num_src_views, N_points),
                        out_nocs=False,
                    )

                if self.density_noise != 0.0 and randomized:
                    raw_sigma = (
                        raw_sigma + torch.rand_like(raw_sigma) * self.density_noise
                    )

                raw_rgb = raw_rgb.reshape(B_fg, N_samples, -1)
                raw_sigma = raw_sigma.reshape(B_fg, N_samples, -1)

                if out_nocs:
                    raw_nocs = raw_nocs.reshape(B_fg, N_samples, -1)

                density_bias = -1.0
                sigma = self.sigma_activation(raw_sigma + density_bias)

                rgb = self.rgb_activation(raw_rgb)
                rgb_padding = 0.001
                rgb = rgb * (1 + 2 * rgb_padding) - rgb_padding

                # Don't apply sigma activation here: First we want to set the opacity of backgrounds to zero where foreground alpha exists
                # sigma = self.sigma_activation(raw_sigma)
                if out_nocs:
                    nocs = self.nocs_activation(raw_nocs)
                    # return rgb, sigma, nocs
                    return rgb, sigma, nocs
                else:
                    # return rgb, sigma
                    return rgb, sigma

            B, N_samples, _ = fg_samples.shape

            world_latent_fg = index_grid(
                fg_samples,
                scene_grid_xz,
                scene_grid_xy,
                scene_grid_yz,
                rays["src_poses"],
                src_views_num=self.num_src_views,
            )
            world_latent_bg = index_grid(
                bg_samples_linear,
                scene_grid_xz,
                scene_grid_xy,
                scene_grid_yz,
                rays["src_poses"],
                src_views_num=self.num_src_views,
            )

            # latent = index_grid(samples, scene_grid_xz, scene_grid_xy, scene_grid_yz)
            # world_latent_fg = latent.squeeze(0)[:,:T_samples].permute(1,0).reshape(B, N_samples, -1)
            # world_latent_bg = latent.squeeze(0)[:,T_samples:].permute(1,0).reshape(B, N_samples, -1)

            # get_color_feats

            local_latent_fg, _ = self.get_local_feats(
                fg_samples,
                rays["src_poses"],
                rays["src_focal"],
                rays["src_c"],
                src_views_num=self.num_src_views,
            )
            local_latent_bg, _ = self.get_local_feats(
                bg_samples_linear[:, :, :3],
                rays["src_poses"],
                rays["src_focal"],
                rays["src_c"],
                src_views_num=self.num_src_views,
            )

            fg_samples_pts = fg_samples[:, :, :3].reshape(-1, 3).unsqueeze(0)
            fg_samples_cam = world2camera(
                fg_samples_pts, rays["src_poses"], self.num_src_views
            )

            bg_samples_pts = bg_samples[:, :, :3].reshape(-1, 3).unsqueeze(0)
            bg_samples_cam = world2camera(
                bg_samples_pts, rays["src_poses"], self.num_src_views
            )
            depth = (
                bg_samples[:, :, 3]
                .view(-1, 1)
                .unsqueeze(0)
                .repeat(bg_samples_cam.shape[0], 1, 1)
            )
            bg_samples_cam = torch.cat((bg_samples_cam, depth), dim=-1)

            # obj_rgb, obj_sigma, nocs_obj = predict(fg_samples_cam, obj_mlp, world_latent_fg, local_latent_fg, B, N_samples, out_nocs = True)
            fg_rgb, fg_sigma = predict(
                fg_samples_cam, fg_mlp, world_latent_fg, local_latent_fg, B, N_samples
            )
            bg_rgb, bg_sigma = predict(
                bg_samples_cam, bg_mlp, world_latent_bg, local_latent_bg, B, N_samples
            )

            if out_depth:
                # obj_comp_rgb, obj_acc, obj_weights, bg_lambda_obj, obj_nocs, obj_depth = helper.volumetric_rendering(
                #     obj_rgb,
                #     obj_sigma,
                #     fg_t_vals,
                #     rays["rays_d"],
                #     in_sphere=True,
                #     t_far=far,
                #     nocs = nocs_obj,
                #     out_depth = True,
                #     white_bkgd = True
                # )

                (
                    fg_comp_rgb,
                    fg_acc,
                    fg_weights,
                    bg_lambda,
                    fg_depth,
                ) = helper.volumetric_rendering(
                    fg_rgb,
                    fg_sigma,
                    fg_t_vals,
                    rays["rays_d"],
                    in_sphere=True,
                    t_far=far,
                    out_depth=True,
                    white_bkgd=False,
                )

                # set the density of rays within fg to zero in bg
                # bg_sigma[~bg_lambda] = 1e-5
                (
                    bg_comp_rgb,
                    bg_acc,
                    bg_weights,
                    _,
                    bg_depth,
                ) = helper.volumetric_rendering(
                    bg_rgb,
                    bg_sigma,
                    bg_t_vals,
                    rays["rays_d"],
                    in_sphere=False,
                    out_depth=True,
                    white_bkgd=False,
                )
                # comp_rgb = obj_comp_rgb + fg_comp_rgb + bg_lambda * bg_comp_rgb
                comp_rgb = fg_comp_rgb + bg_lambda * bg_comp_rgb
                comp_depth = fg_depth + bg_lambda.squeeze(-1) * bg_depth
                # ret.append((comp_rgb, fg_comp_rgb, bg_comp_rgb, obj_comp_rgb, fg_acc, bg_lambda, obj_acc, obj_nocs, comp_depth))
                ret.append(
                    (comp_rgb, fg_comp_rgb, bg_comp_rgb, fg_acc, bg_lambda, comp_depth)
                )
            else:
                # obj_comp_rgb, obj_acc, obj_weights, bg_lambda_obj, obj_nocs = helper.volumetric_rendering(
                #     obj_rgb,
                #     obj_sigma,
                #     fg_t_vals,
                #     rays["rays_d"],
                #     white_bkgd=white_bkgd,
                #     in_sphere=True,
                #     t_far=far,
                #     # t_far=far_obj,
                #     nocs = nocs_obj
                # )

                (
                    fg_comp_rgb,
                    fg_acc,
                    fg_weights,
                    bg_lambda,
                ) = helper.volumetric_rendering(
                    fg_rgb,
                    fg_sigma,
                    fg_t_vals,
                    rays["rays_d"],
                    white_bkgd=white_bkgd,
                    in_sphere=True,
                    t_far=far,
                )
                bg_comp_rgb, bg_acc, bg_weights, _ = helper.volumetric_rendering(
                    bg_rgb,
                    bg_sigma,
                    bg_t_vals,
                    rays["rays_d"],
                    white_bkgd=white_bkgd,
                    in_sphere=False,
                )

                fg_sdist = 0.5 * (fg_t_vals[..., 1:] + fg_t_vals[..., :-1])
                diff = fg_sdist[:, -1] - fg_sdist[:, -2]
                last_val = fg_sdist[:, -1] + diff
                fg_sdist = torch.cat([fg_sdist, last_val.unsqueeze(-1)], dim=-1)

                bg_sdist = 0.5 * (bg_t_vals[..., 1:] + bg_t_vals[..., :-1])
                bg_sdist = torch.cat(
                    [bg_sdist, bg_t_vals[..., -1].unsqueeze(-1)], dim=-1
                )
                # comp_rgb = obj_comp_rgb + fg_comp_rgb + bg_lambda * bg_comp_rgb
                comp_rgb = fg_comp_rgb + bg_lambda * bg_comp_rgb
                # ret.append((comp_rgb, fg_comp_rgb, bg_comp_rgb, obj_comp_rgb, fg_acc, bg_lambda, obj_acc, obj_nocs, fg_weights, bg_weights, fg_sdist, bg_sdist))
                # ret.append((comp_rgb, obj_comp_rgb, obj_acc, obj_nocs, fg_weights, bg_weights, fg_sdist, bg_sdist, bg_acc))
                ret.append(
                    (comp_rgb, fg_weights, bg_weights, fg_sdist, bg_sdist, bg_acc)
                )

        return ret


# @gin.configurable()
class LitNeRFTP_FUSION_CONV_SCENE(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
        grad_max_norm: float = 0.05,
    ):
        self.save_hyperparameters(hparams)
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        if type(hparams) is dict:
            hparams = DotMap(hparams)
        self.hparams.update(vars(hparams))
        super(LitNeRFTP_FUSION_CONV_SCENE, self).__init__()

        eval_inference = self.hparams.render_name
        if eval_inference is not None:
            num = int(eval_inference[0])
            self.model = NeRF_TP(num_src_views=num)
        elif self.hparams.is_optimize is not None:
            num = int(self.hparams.is_optimize[0])
            self.model = NeRF_TP(
                num_src_views=num, is_optimize=self.hparams.is_optimize
            )
        else:
            self.model = NeRF_TP()

        if self.hparams.finetune_lpips:
            self.loss_fn = None
        elif (
            self.hparams.render_name is not None and "LPIPS" in self.hparams.render_name
        ):
            self.loss_fn = lpips.LPIPS(net="vgg")

        # self.loss_fn = lpips.LPIPS(net='alex')

    def on_train_epoch_start(self):
        if self.hparams.finetune_lpips:
            if self.loss_fn is None:
                self.loss_fn = lpips.LPIPS(net="vgg")

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]

        if (self.hparams.dataset_name == "pd_multi_obj"
            or self.hparams.dataset_name == "pd_multi_obj_ae"
        ):
            kwargs_train = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "nerfpp",
                "optimize": self.hparams.is_optimize,
                "encoder_type": self.hparams.encoder_type,
                "contract": False,
                "finetune_lpips": self.hparams.finetune_lpips,
            }
            kwargs_val = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "nerfpp",
                "optimize": self.hparams.is_optimize,
                "encoder_type": self.hparams.encoder_type,
                "contract": False,
                "finetune_lpips": self.hparams.finetune_lpips,
            }

        if self.hparams.eval_mode is not None:
            kwargs_test = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "nerfpp",
                "eval_inference": self.hparams.render_name,
                "encoder_type": self.hparams.encoder_type,
                "contract": False,
                "finetune_lpips": self.hparams.finetune_lpips,
            }

            # 'eval_inference': None}
            if self.hparams.eval_mode == "full_eval":
                split = "val"
            else:
                split = "test"
            self.test_dataset = dataset(split=split, **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back
        # if torch.is_tensor(self.train_dataset.xyz_min):
        #     xyz_min = self.train_dataset.xyz_min
        #     xyz_max = self.train_dataset.xyz_max
        # else:
        #     xyz_min = torch.from_numpy(self.train_dataset.xyz_min)
        #     xyz_max = torch.from_numpy(self.train_dataset.xyz_max)

        # xyz_min = None
        # xyz_max = None

    def training_step(self, batch, batch_idx):
        # eps = 1e-6

        for k, v in batch.items():
            batch[k] = v.squeeze(0)

        # self.model.encode(batch["src_imgs"], batch["src_poses"], batch["src_focal"], batch["src_c"])

        # print("=====================================================\n\n\n")
        # for n, p in self.model.named_parameters():
        #     if p.grad is None:
        #         print(f'{n} has no grad')
        # print("=======================================================\n\n\n")

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        #     rendered_results = self.model(
        #         batch, self.randomized, self.white_bkgd, self.near, self.far, out_depth=False
        #     )

        # prof.export_chrome_trace("trace.json")

        # with profiler.profile(with_stack=True, profile_memory=True, record_shapes=True) as prof:
        #     rendered_results = self.model(
        #         batch, self.randomized, self.white_bkgd, self.near, self.far, out_depth=False
        #     )

        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_memory_usage', row_limit=10))

        rendered_results = self.model(
            batch,
            self.randomized,
            self.white_bkgd,
            self.near,
            self.far,
            out_depth=False,
        )

        # obj_rgb_coarse = rendered_results[0][1]
        # obj_rgb_fine = rendered_results[1][1]

        # nocs_coarse = rendered_results[0][3]
        # nocs_fine = rendered_results[1][3]

        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]

        target = batch["target"]
        # target_nocs = batch["nocs_2d"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
        loss = loss1 + loss0

        if self.hparams.finetune_lpips:
            loss_lpips0 = self.lpips_loss(rgb_coarse, target)
            loss_lpips1 = self.lpips_loss(rgb_fine, target)
            lpips_loss = loss_lpips0 + loss_lpips1
            self.log("train/lpips_loss", lpips_loss, on_step=True)
            loss += lpips_loss

        # if loss.isnan(): loss=eps
        # else: loss = loss

        # mask = batch["instance_mask"].view(-1, 1).repeat(1, 3)

        # obj_rgb_coarse[~mask] = 0
        # obj_rgb_fine[~mask] = 0

        # if torch.any(mask > 0):
        #     loss2 = helper.img2mse(obj_rgb_coarse[mask], target[mask])
        #     loss3 = helper.img2mse(obj_rgb_fine[mask], target[mask])
        #     masked_rgb_loss = (loss2 + loss3)

        #     loss4 = helper.img2mse(nocs_coarse[mask], target_nocs[mask])
        #     loss5 = helper.img2mse(nocs_fine[mask], target_nocs[mask])
        #     masked_nocs_loss = (loss4 + loss5)
        #     self.log("train/masked_rgb_loss", masked_rgb_loss, on_step=True)
        #     self.log("train/masked_nocs_loss", masked_nocs_loss, on_step=True)
        #     # loss += masked_rgb_loss

        #     loss += masked_rgb_loss
        #     loss += masked_nocs_loss

        # if masked_nocs_loss.isnan(): masked_nocs_loss=eps
        # else: masked_nocs_loss = masked_nocs_loss

        # opacity_reg_loss = self.opacity_regularization_loss(rendered_results)

        # self.log("train/opacity_reg_loss", opacity_reg_loss, on_step=True)
        # loss += opacity_reg_loss

        # opacity_reg_bg = self.opacity_regularization_loss_bg(rendered_results)
        # self.log("train/opacity_reg_bg_loss", opacity_reg_bg, on_step=True)
        # loss += opacity_reg_bg

        # opacity loss
        # opacity_loss = self.opacity_loss(
        #         rendered_results, batch["instance_mask"].view(-1)
        #     )
        # self.log("train/opacity loss", opacity_loss, on_step=True)
        # loss += opacity_loss

        dist_loss = self.distortion_loss(rendered_results)
        loss += dist_loss
        self.log("train/dist loss", dist_loss, on_step=True)

        # interlevel_loss = self.interlevel_loss(rendered_results)
        # loss += interlevel_loss
        # self.log("train/interlevel_loss", interlevel_loss, on_step=True)

        # if opacity_loss.isnan(): opacity_loss=eps
        # else: opacity_loss = opacity_loss

        # We might not need opacity loss if we are supplying object masks
        # self.log("train/sem_map_loss", opacity_loss, on_step=True)

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)

        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True)
        self.log("train/lr", helper.get_learning_rate(self.optimizers()))
        return loss

    def render_rays(self, batch):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if (
                    k == "src_imgs"
                    or k == "src_poses"
                    or k == "src_focal"
                    or k == "src_c"
                ):
                    batch_chunk[k] = v
                elif k == "radii":
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]

            # do not suppress rays for near background mlp in validation since we don't have masks in inference time
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far, out_depth=True
            )

            # ret["obj_acc"] +=[rendered_results_chunk[1][6]]
            ret["comp_rgb"] += [rendered_results_chunk[1][0]]
            ret["fg_rgb"] += [rendered_results_chunk[1][1]]
            ret["bg_rgb"] += [rendered_results_chunk[1][2]]
            ret["obj_rgb"] += [rendered_results_chunk[1][3]]
            # ret["nocs"] +=[rendered_results_chunk[1][7]]

            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        self.log("val/psnr", psnr_.item(), on_step=True, prog_bar=True, logger=True)
        return ret

    def render_rays_test(self, batch):
        # for k,v in batch.items():
        #     print(k,v.shape)
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if (
                    k == "src_imgs"
                    or k == "src_poses"
                    or k == "src_focal"
                    or k == "src_c"
                ):
                    batch_chunk[k] = v
                elif k == "radii":
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]

            # do not suppress rays for near background mlp in validation since we don't have masks in inference time
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far, out_depth=True
            )
            # here 1 denotes fine
            ret["comp_rgb"] += [rendered_results_chunk[1][0]]
            ret["fg_rgb"] += [rendered_results_chunk[1][1]]
            ret["bg_rgb"] += [rendered_results_chunk[1][2]]
            # ret["obj_rgb"] +=[rendered_results_chunk[1][3]]
            ret["depth"] += [rendered_results_chunk[1][5]]
            # ret["obj_acc"] +=[rendered_results_chunk[1][6]]
            # ret["nocs"] +=[rendered_results_chunk[1][7]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        test_output = {}
        test_output["target"] = batch["target"]
        test_output["rgb"] = ret["comp_rgb"]
        # test_output["obj_rgb"] = ret["obj_rgb"]
        test_output["depth"] = ret["depth"]
        # test_output["obj_nocs"] = ret["nocs"]
        # test_output["obj_acc"] = ret["obj_acc"]
        test_output["instance_mask"] = batch["instance_mask"]
        # print("ret[comp_rgb], ret[comp_rgb]", ret["comp_rgb"].shape, ret["depth"].shape, ret["obj_rgb"].shape)
        return test_output

    # def render_rays(self, batch, batch_idx):
    #     ret = {}
    #     rendered_results = self.model(
    #         batch, False, self.white_bkgd, self.near, self.far
    #     )
    #     rgb_fine = rendered_results[1][0]
    #     target = batch["target"]
    #     ret["target"] = target
    #     ret["rgb"] = rgb_fine
    #     return ret

    def on_validation_start(self):
        self.random_batch = np.random.randint(5, size=1)[0]

    def validation_step(self, batch, batch_idx):
        for k, v in batch.items():
            batch[k] = v.squeeze(0)
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)

        for k, v in batch.items():
            print(k, v.shape)

        # self.model.encode(batch["src_imgs"], batch["src_poses"], batch["src_focal"], batch["src_c"])
        W, H = self.hparams.img_wh
        ret = self.render_rays(batch)
        # rank = dist.get_rank()
        rank = 0
        if rank == 0:
            if batch_idx == self.random_batch:
                grid_img = visualize_val_fb_bg_rgb((W, H), batch, ret)
                self.logger.experiment.log({"val/GT_pred rgb": wandb.Image(grid_img)})

    def test_step(self, batch, batch_idx):
        for k, v in batch.items():
            batch[k] = v.squeeze(0)
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)
        # for k, v in batch.items():
        #     print(k, v.shape)
        # self.model.encode(batch["src_imgs"], batch["src_poses"], batch["src_focal"], batch["src_c"])
        ret = self.render_rays_test(batch)
        return ret

    def configure_optimizers(self):
        if self.hparams.is_optimize is not None or self.hparams.finetune_lpips:
            print("HEREEE,\n\n\n")
            lr_init = 5.0e-6

            optimizer = torch.optim.Adam(
                params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
            )

            if self.trainer.resume_from_checkpoint is not None:
                optimizer.param_groups[0]["lr"] = lr_init

            # Freeze the weights of the spatial_encoder module
            for param in self.model.encoder.spatial_encoder.parameters():
                param.requires_grad = False

            # Set the mode of the spatial_encoder module to evaluation
            self.model.encoder.spatial_encoder.eval()

            # module.eval() for batchnorm layers of triplanar convs
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

            return optimizer

        else:
            return torch.optim.Adam(
                params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
            )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        if self.hparams.is_optimize:
            optimizer.step(closure=optimizer_closure)

        else:
            step = self.trainer.global_step
            max_steps = self.hparams.run_max_steps

            if self.lr_delay_steps > 0:
                delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0

            t = np.clip(step / max_steps, 0, 1)
            scaled_lr = np.exp(
                np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t
            )
            new_lr = delay_rate * scaled_lr

            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

            if self.grad_max_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), self.grad_max_norm)

            optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=32,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,
            pin_memory=True,
        )

    # def validation_epoch_end(self, outputs):
    #     val_image_sizes = self.val_dataset.val_image_sizes
    #     rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
    #     targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
    #     psnr_mean = self.psnr_each(rgbs, targets).mean()
    #     ssim_mean = self.ssim_each(rgbs, targets).mean()
    #     lpips_mean = self.lpips_each(rgbs, targets).mean()
    #     self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        # dmodule = self.trainer.datamodule
        # all_image_sizes = (
        #     dmodule.all_image_sizes
        #     if not dmodule.eval_test_only
        #     else dmodule.test_image_sizes
        # )
        all_image_sizes = self.test_dataset.image_sizes
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        instance_masks = self.alter_gather_cat(
            outputs, "instance_mask", all_image_sizes
        )
        # accs = self.alter_gather_cat(outputs, "obj_acc", all_image_sizes)
        # obj_rgbs = self.alter_gather_cat(outputs, "obj_rgb", all_image_sizes)
        # obj_nocs = self.alter_gather_cat(outputs, "obj_nocs", all_image_sizes)
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)

        all_obj_rgbs, all_target_rgbs = get_obj_rgbs_from_segmap(
            instance_masks, rgbs, targets
        )

        depths = self.alter_gather_cat(outputs, "depth", all_image_sizes)
        # psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # lpips = self.lpips(
        #     rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        # )

        if self.hparams.eval_mode == "full_eval":
            psnr = self.psnr(rgbs, targets, None, None, None)
            ssim = self.ssim(rgbs, targets, None, None, None)
            lpips = self.lpips(rgbs, targets, None, None, None)
            print("psnr, ssim, lpips", psnr, ssim, lpips)

            psnr_obj = self.psnr(all_obj_rgbs, all_target_rgbs, None, None, None)
            print("psnr obj", psnr_obj)

            self.log("test/psnr", psnr["test"], on_epoch=True)
            self.log("test/ssim", ssim["test"], on_epoch=True)
            self.log("test/lpips", lpips["test"], on_epoch=True)
            self.log("test/psnr_obj", psnr_obj["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(
                "ckpts", self.hparams.exp_name, self.hparams.render_name
            )
            os.makedirs(image_dir, exist_ok=True)
            store_image(image_dir, rgbs, "image")

            # image_dir = os.path.join("ckpts",self.hparams.exp_name, self.hparams.render_name)
            # os.makedirs(image_dir, exist_ok=True)
            # store_image(image_dir, accs, 'segmentation')

            # image_dir = os.path.join("ckpts",self.hparams.exp_name, self.hparams.render_name)
            # os.makedirs(image_dir, exist_ok=True)
            # store_image(image_dir, obj_rgbs, 'obj')

            # image_dir = os.path.join("ckpts",self.hparams.exp_name, self.hparams.render_name)
            # os.makedirs(image_dir, exist_ok=True)
            # store_image(image_dir, obj_nocs, 'nocs')

            image_dir = os.path.join(
                "ckpts", self.hparams.exp_name, self.hparams.render_name
            )
            os.makedirs(image_dir, exist_ok=True)
            store_depth_img(image_dir, depths, "depth_img")

            image_dir = os.path.join(
                "ckpts", self.hparams.exp_name, self.hparams.render_name
            )
            os.makedirs(image_dir, exist_ok=True)
            store_depth_raw(image_dir, depths, "depth_raw")

            if self.hparams.eval_mode == "full_eval":
                result_path = os.path.join(
                    "ckpts", self.hparams.exp_name, "results.json"
                )
                write_stats(result_path, psnr, ssim, lpips)

        # return psnr, ssim, lpips

    # def opacity_loss(self, rendered_results, instance_mask):
    #     # opacity_lambda = 1.0

    #     #0.2 for this run works nicely https://wandb.ai/tri/generalizable-scene-representations/runs/1qh068ug?workspace=user-zirshad-tri
    #     opacity_lambda = 0.2
    #     criterion = nn.MSELoss(reduction="none")
    #     loss = (
    #         criterion(
    #             torch.clamp(rendered_results[0][2], 0, 1),
    #             instance_mask.float(),
    #         )
    #     ).mean()
    #     loss += (
    #         criterion(
    #             torch.clamp(rendered_results[1][2], 0, 1),
    #             instance_mask.float(),
    #         )
    #     ).mean()
    #     #
    #     return loss*opacity_lambda

    # def opacity_loss_CE(self, rendered_results, instance_mask):
    #     opacity_lambda = 0.2
    #     criterion = nn.BCEWithLogitsLoss()
    #     loss = (
    #         criterion(
    #             torch.clamp(rendered_results[0][6], 0, 1),
    #             instance_mask.float(),
    #         )
    #     ).mean()
    #     loss += (
    #         criterion(
    #             torch.clamp(rendered_results[1][6], 0, 1),
    #             instance_mask.float(),
    #         )
    #     ).mean()
    #     #
    #     return loss*opacity_lambda
    #     # return loss

    # def opacity_regularization_loss(self, rendered_results):
    #     opacity_lambda = 0.05
    #     loss_opacity = (torch.clamp(rendered_results[0][2], 0, 1) ** 2).mean()
    #     loss_opacity += (torch.clamp(rendered_results[1][2], 0, 1) ** 2).mean()
    #     return opacity_lambda* loss_opacity

    # def opacity_regularization_loss(self, rendered_results):
    #     opacity_lambda = 0.05
    #     alpha_coarse = rendered_results[0][2].clamp(1e-5, 1 - 1e-5)
    #     alpha_fine = rendered_results[1][2].clamp(1e-5, 1 - 1e-5)

    #     loss_entropy = (- alpha_coarse * torch.log2(alpha_coarse) - (1 - alpha_coarse) * torch.log2(1 - alpha_coarse)).mean()
    #     loss_entropy += (- alpha_fine * torch.log2(alpha_fine) - (1 - alpha_fine) * torch.log2(1 - alpha_fine)).mean()
    #     # loss_opacity = (torch.clamp(rendered_results[0][2], 0, 1) ** 2).mean()
    #     # loss_opacity += (torch.clamp(rendered_results[1][2], 0, 1) ** 2).mean()
    #     return opacity_lambda* loss_entropy

    # def opacity_regularization_loss_bg(self, rendered_results):
    #     opacity_lambda = 0.05
    #     alpha_coarse = rendered_results[0][8].clamp(1e-5, 1 - 1e-5)
    #     alpha_fine = rendered_results[1][8].clamp(1e-5, 1 - 1e-5)

    #     loss_entropy = (- alpha_coarse * torch.log2(alpha_coarse) - (1 - alpha_coarse) * torch.log2(1 - alpha_coarse)).mean()
    #     loss_entropy += (- alpha_fine * torch.log2(alpha_fine) - (1 - alpha_fine) * torch.log2(1 - alpha_fine)).mean()
    #     # loss_opacity = (torch.clamp(rendered_results[0][2], 0, 1) ** 2).mean()
    #     # loss_opacity += (torch.clamp(rendered_results[1][2], 0, 1) ** 2).mean()
    #     return opacity_lambda* loss_entropy

    # def interlevel_loss(self, rendered_results):
    #     last_ray_results = rendered_results[-1]

    #     c_fg = last_ray_results[1][6].detach()
    #     w_fg = last_ray_results[1][4].detach()

    #     c_bg = last_ray_results[1][7].detach()
    #     w_bg = last_ray_results[1][5].detach()

    #     loss_interlevel = 0.0
    #     for ray_results in rendered_results[:-1]:
    #         cp_fg = ray_results[1][6]
    #         wp_fg = ray_results[1][4]

    #         cp_bg = ray_results[1][7]
    #         wp_bg = ray_results[1][5]

    #         loss_interlevel += torch.mean(helper.lossfun_outer(c_fg, w_fg, cp_fg, wp_fg))
    #         loss_interlevel += torch.mean(helper.lossfun_outer(c_bg, w_bg, cp_bg, wp_bg))
    #     return loss_interlevel

    # def distortion_loss(self, rendered_results):
    #     last_ray_results = rendered_results[-1]
    #     c_fg = last_ray_results[1][6]
    #     w_fg = last_ray_results[1][4]

    #     c_bg = last_ray_results[1][7]
    #     w_bg = last_ray_results[1][5]
    #     loss = 0.01*torch.mean(helper.lossfun_distortion(c_fg, w_fg))
    #     loss += 0.01*torch.mean(helper.lossfun_distortion(c_bg, w_bg))
    #     return loss

    def distortion_loss(self, rendered_results):
        fg_w = rendered_results[1][1]
        bg_w = rendered_results[1][2]
        m_fg = rendered_results[1][3]
        m_bg = rendered_results[1][4]
        _, N = fg_w.shape
        interval = 1 / N

        # norm the weights
        # fg_w_clone = fg_w.clone() / fg_w.sum(-1, keepdim=True)
        # bg_w_clone = bg_w.clone() / bg_w.sum(-1, keepdim=True)

        loss = 0.01 * eff_distloss(fg_w, m_fg, interval)
        loss += 0.01 * eff_distloss(bg_w, m_bg, interval)
        return loss

    # def surface_loss(self, w):
    #     p = torch.exp(-torch.abs(w)) + torch.exp(-torch.abs(1 - w))
    #     # Calculate the log probability of each weight value
    #     log_p = torch.log(p)
    #     # Sum the log probabilities to get the total loss
    #     loss = -torch.sum(log_p)
    #     return loss

    # def hard_surface_loss(self, rendered_results):
    #     lambda_surface = 0.1
    #     # Calculate the probability of each weight value using the given formula
    #     loss = lambda_surface* self.surface_loss(rendered_results[0][8])
    #     loss += lambda_surface * self.surface_loss(rendered_results[1][8])
    #     loss += lambda_surface * self.surface_loss(rendered_results[0][9])
    #     loss += lambda_surface * self.surface_loss(rendered_results[1][9])
    #     loss += lambda_surface * self.surface_loss(rendered_results[0][10])
    #     loss += lambda_surface * self.surface_loss(rendered_results[1][10])
    #     # Multiply the loss by the surface loss weight
    #     loss *= 0.1
    #     return loss

    def lpips_loss(self, pred_rgb, gt_rgb):
        self.loss_fn.to(pred_rgb.device)

        gt_rgb_scaled = (
            2
            * (
                torch.clamp(gt_rgb, 0, 1)
                .view(30, 30, 3)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            - 1
        )
        pred_rgb_scaled = (
            2
            * (
                torch.clamp(pred_rgb, 0, 1)
                .view(30, 30, 3)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            - 1
        )

        d = self.loss_fn.forward(pred_rgb_scaled, gt_rgb_scaled)
        lpips_lambda = 0.3
        return d.squeeze() * lpips_lambda
