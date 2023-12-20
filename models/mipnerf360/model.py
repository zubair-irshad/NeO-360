# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Mip-NeRF360 (https://github.com/google-research/multinerf)
# Copyright (c) 2022 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import models.mipnerf360.helper as helper
from models.utils import store_image, write_stats, get_obj_rgbs_from_segmap
from PIL import Image
from models.interface import LitModel
from datasets import dataset_dict
from torch.utils.data import DataLoader
import wandb
from collections import defaultdict
from utils.train_helper import *


# @gin.configurable()
class MipNeRF360MLP(nn.Module):
    def __init__(
        self,
        netdepth: int = 8,
        netwidth: int = 256,
        bottleneck_width: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        min_deg_point: int = 0,
        max_deg_point: int = 12,
        skip_layer: int = 4,
        skip_layer_dir: int = 4,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
        deg_view: int = 4,
        bottleneck_noise: float = 0.0,
        density_bias: float = -1.0,
        density_noise: float = 0.0,
        rgb_premultiplier: float = 1.0,
        rgb_bias: float = 0.0,
        rgb_padding: float = 0.001,
        basis_shape: str = "icosahedron",
        basis_subdivision: int = 2,
        disable_rgb: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(MipNeRF360MLP, self).__init__()

        self.net_activation = nn.ReLU()
        self.density_activation = nn.Softplus()
        self.rgb_activation = nn.Sigmoid()
        self.warp_fn = helper.contract
        self.register_buffer(
            "pos_basis_t", helper.generate_basis(basis_shape, basis_subdivision)
        )

        pos_size = ((max_deg_point - min_deg_point) * 2) * self.pos_basis_t.shape[-1]
        view_pos_size = (deg_view * 2 + 1) * 3

        module = nn.Linear(pos_size, netwidth)
        init.kaiming_uniform_(module.weight)
        pts_linear = [module]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.kaiming_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linear = nn.ModuleList(pts_linear)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        init.kaiming_uniform_(self.density_layer.weight)

        if not disable_rgb:
            self.bottleneck_layer = nn.Linear(netwidth, bottleneck_width)
            layer = nn.Linear(bottleneck_width + view_pos_size, netwidth_condition)
            init.kaiming_uniform_(layer.weight)
            views_linear = [layer]
            for idx in range(netdepth_condition - 1):
                if idx % skip_layer_dir == 0 and idx > 0:
                    layer = nn.Linear(
                        netwidth_condition + view_pos_size, netwidth_condition
                    )
                else:
                    layer = nn.Linear(netwidth_condition, netwidth_condition)
                init.kaiming_uniform_(layer.weight)
                views_linear.append(layer)
            self.views_linear = nn.ModuleList(views_linear)

            self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

            init.kaiming_uniform_(self.bottleneck_layer.weight)
            init.kaiming_uniform_(self.rgb_layer.weight)

        self.dir_enc_fn = helper.pos_enc

    def predict_density(self, means, covs, randomized, is_train):
        means, covs = self.warp_fn(means, covs, is_train)

        lifted_means, lifted_vars = helper.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )
        x = helper.integrated_pos_enc(
            lifted_means, lifted_vars, self.min_deg_point, self.max_deg_point
        )

        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linear[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x)[..., 0]
        if self.density_noise > 0.0 and randomized:
            raw_density += self.density_noise * torch.rand_like(raw_density)

        return raw_density, x

    def forward(self, gaussians, viewdirs, randomized, is_train):
        means, covs = gaussians

        raw_density, x = self.predict_density(means, covs, randomized, is_train)
        density = self.density_activation(raw_density + self.density_bias)

        if self.disable_rgb:
            rgb = torch.zeros_like(means)
            return {
                "density": density,
                "rgb": rgb,
            }

        bottleneck = self.bottleneck_layer(x)
        if self.bottleneck_noise > 0.0 and randomized:
            bottleneck += torch.rand_like(bottleneck) * self.bottleneck_noise
        x = [bottleneck]

        dir_enc = self.dir_enc_fn(viewdirs, 0, self.deg_view, True)
        dir_enc = torch.broadcast_to(
            dir_enc[..., None, :], bottleneck.shape[:-1] + (dir_enc.shape[-1],)
        )
        x.append(dir_enc)
        x = torch.cat(x, dim=-1)

        inputs = x
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer_dir == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        x = self.rgb_layer(x)
        rgb = self.rgb_activation(self.rgb_premultiplier * x + self.rgb_bias)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return {
            "density": density,
            "rgb": rgb,
        }


# @gin.configurable()
class NeRFMLP(MipNeRF360MLP):
    def __init__(
        self,
        netdepth: int = 8,
        netwidth: int = 1024,
    ):
        super(NeRFMLP, self).__init__(netdepth=netdepth, netwidth=netwidth)


# @gin.configurable()
class PropMLP(MipNeRF360MLP):
    def __init__(
        self,
        netdepth: int = 4,
        netwidth: int = 256,
    ):
        super(PropMLP, self).__init__(
            netdepth=netdepth, netwidth=netwidth, disable_rgb=True
        )


# @gin.configurable()
class MipNeRF360(nn.Module):
    def __init__(
        self,
        num_prop_samples: int = 64,
        num_nerf_samples: int = 32,
        num_levels: int = 3,
        bg_intensity_range: Tuple[float] = (1.0, 1.0),
        anneal_slope: int = 10,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        ray_shape: str = "cone",
        disable_integration: bool = False,
        single_jitter: bool = True,
        dilation_multiplier: float = 0.5,
        dilation_bias: float = 0.0025,
        num_glo_features: int = 0,
        num_glo_embeddings: int = 1000,
        learned_exposure_scaling: bool = False,
        near_anneal_rate: Optional[float] = None,
        near_anneal_init: float = 0.95,
        single_mlp: bool = False,
        resample_padding: float = 0.0,
        use_gpu_resampling: bool = False,
        opaque_background: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(MipNeRF360, self).__init__()
        self.mlps = nn.ModuleList(
            [PropMLP() for _ in range(num_levels - 1)]
            + [
                NeRFMLP(),
            ]
        )

    def forward(self, batch, train_frac, randomized, is_train, near, far):
        bsz, _ = batch["rays_o"].shape
        device = batch["rays_o"].device

        _, s_to_t = helper.construct_ray_warps(near, far)
        if self.near_anneal_rate is None:
            init_s_near = 0.0
        else:
            init_s_near = 1 - train_frac / self.near_anneal_rate
            init_s_near = max(min(init_s_near, 1), 0)
        init_s_far = 1.0

        sdist = torch.cat(
            [
                torch.full((bsz, 1), init_s_near, device=device),
                torch.full((bsz, 1), init_s_far, device=device),
            ],
            dim=-1,
        )

        weights = torch.ones(bsz, 1, device=device)
        prod_num_samples = 1

        ray_history = []
        renderings = []

        for i_level in range(self.num_levels):
            is_prop = i_level < (self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples

            dilation = (
                self.dilation_bias
                + self.dilation_multiplier
                * (init_s_far - init_s_near)
                / prod_num_samples
            )

            prod_num_samples *= num_samples

            use_dilation = self.dilation_bias > 0 or self.dilation_multiplier > 0

            if i_level > 0 and use_dilation:
                sdist, weights = helper.max_dilate_weights(
                    sdist,
                    weights,
                    dilation,
                    domain=(init_s_near, init_s_far),
                    renormalize=True,
                )
                sdist = sdist[..., 1:-1]
                weights = weights[..., 1:-1]

            if self.anneal_slope > 0:
                bias = lambda x, s: (s * x) / ((s - 1) * x + 1)
                anneal = bias(train_frac, self.anneal_slope)
            else:
                anneal = 1.0

            logits_resample = torch.where(
                sdist[..., 1:] > sdist[..., :-1],
                anneal * torch.log(weights + self.resample_padding),
                torch.full_like(weights, -torch.inf),
            )

            sdist = helper.sample_intervals(
                randomized,
                sdist,
                logits_resample,
                num_samples,
                single_jitter=self.single_jitter,
                domain=(init_s_near, init_s_far),
            )

            if self.stop_level_grad:
                sdist = sdist.detach()

            tdist = s_to_t(sdist)

            gaussians = helper.cast_rays(
                tdist,
                batch["rays_o"],
                batch["rays_d"],
                batch["radii"],
                self.ray_shape,
                diag=False,
            )

            if self.disable_integration:
                gaussians = (gaussians[0], torch.zeros_like(gaussians[1]))

            ray_results = self.mlps[i_level](
                gaussians, batch["viewdirs"], randomized, is_train
            )

            weights = helper.compute_alpha_weights(
                ray_results["density"],
                tdist,
                batch["rays_d"],
                opaque_background=self.opaque_background,
            )[0]

            if self.bg_intensity_range[0] == self.bg_intensity_range[1]:
                bg_rgbs = self.bg_intensity_range[0]
            elif not randomized:
                bg_rgbs = (
                    self.bg_intensity_range[0] + self.bg_intensity_range[1]
                ) / 2.0
            else:
                bg_rgbs = (
                    torch.rand(3)
                    * (self.bg_intensity_range[1] - self.bg_intensity_range[0])
                    + self.bg_intensity_range[0]
                )

            rendering = helper.volumetric_rendering(
                ray_results["rgb"],
                weights,
                tdist,
                bg_rgbs,
                far,
                False,
            )

            ray_results["sdist"] = sdist
            ray_results["weights"] = weights

            ray_history.append(ray_results)
            renderings.append(rendering)

        return renderings, ray_history


class LitMipNeRF360(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 2.0e-3,
        lr_final: float = 2.0e-5,
        lr_delay_steps: int = 512,
        lr_delay_mult: float = 0.01,
        data_loss_mult: float = 1.0,
        interlevel_loss_mult: float = 1.0,
        distortion_loss_mult: float = 0.01,
        use_multiscale: bool = False,
        charb_padding: float = 0.001,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))

        super(LitMipNeRF360, self).__init__()
        self.model = MipNeRF360()

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]

        kwargs_train = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "mip_nerf",
        }
        kwargs_val = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple((int(self.hparams.img_wh[0]), int(self.hparams.img_wh[1]))),
            "white_back": self.hparams.white_back,
            "model_type": "mip_nerf",
        }

        if self.hparams.run_eval:
            kwargs_test = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "mip_nerf",
                "eval_inference": self.hparams.render_name,
            }
            self.test_dataset = dataset(split="test_val", **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back

    def training_step(self, batch, batch_idx):
        for k, v in batch.items():
            print(k, v.shape)

        for k, v in batch.items():
            if k == "obj_idx" or k == "instance_ids":
                continue
            batch[k] = v.squeeze(0)

        max_steps = 1000000
        train_frac = self.global_step / max_steps
        print("train_func", train_frac)
        rendered_results, ray_history = self.model(
            batch, train_frac, True, True, self.near, self.far
        )
        rgb = rendered_results[-1]["rgb"]
        target = batch["target"]

        rgbloss = helper.img2mse(rgb, target)

        loss = 0.0
        loss = (
            loss + torch.sqrt(rgbloss + self.charb_padding**2) * self.data_loss_mult
        )
        loss = loss + self.interlevel_loss(ray_history) * self.interlevel_loss_mult
        loss = loss + self.distortion_loss(ray_history) * self.distortion_loss_mult

        psnr = helper.mse2psnr(rgbloss)

        self.log("train/loss", loss.item(), on_step=True, prog_bar=True)
        self.log("train/psnr", psnr.item(), on_step=True, prog_bar=True)

        return loss

    # def render_rays(self, batch, batch_idx):
    #     ret = {}
    #     max_steps = self.trainer.max_steps
    #     train_frac = self.global_step / max_steps
    #     rendered_results, ray_history = self.model(
    #         batch, train_frac, False, False, self.near, self.far
    #     )
    #     rgb = rendered_results[-1]["rgb"]
    #     target = batch["target"]
    #     ret["target"] = target
    #     ret["rgb"] = rgb
    #     return ret

    def render_rays_test(self, batch, batch_idx):
        max_steps = self.trainer.max_steps
        train_frac = self.global_step / max_steps
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == "radii":
                    batch_chunk[k] = v[i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]

            for k, v in batch_chunk.items():
                print("batch chunk, k,v", k, v.shape)

            rendered_results_chunk, ray_history = self.model(
                batch_chunk, train_frac, False, False, self.near, self.far
            )
            # rendered_results_chunk = self.model(
            #     batch_chunk, False, self.white_bkgd, self.near, self.far
            # )
            # here 1 denotes fine
            ret["comp_rgb"] += [rendered_results_chunk[-1]["rgb"]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        target = batch["target"]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        test_output = {}
        test_output["target"] = batch["target"]
        test_output["instance_mask"] = batch["instance_mask"]
        test_output["rgb"] = ret["comp_rgb"]
        return test_output

    def render_rays(self, batch, batch_idx):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)

        max_steps = self.trainer.max_steps
        train_frac = self.global_step / max_steps
        print("train_func", train_frac)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == "obj_idx" or k == "instance_ids":
                    batch_chunk[k] = v
                if k == "radii":
                    batch_chunk[k] = v[i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]
            # rendered_results_chunk = self.model(
            #     batch_chunk, False, self.white_bkgd, self.near, self.far
            # )
            for k, v in batch_chunk.items():
                print("batch chunk, k,v", k, v.shape)
            rendered_results_chunk, ray_history = self.model(
                batch_chunk, train_frac, False, False, self.near, self.far
            )
            ret["comp_rgb"] += [rendered_results_chunk[-1]["rgb"]]

        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)
        psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        self.log("val/psnr", psnr_, on_step=True, prog_bar=True, logger=True)
        return ret

    def validation_step(self, batch, batch_idx):
        # for k,v in batch.items():
        #     batch[k] = v.squeeze()
        #     if k =='radii':
        #         batch[k] = batch[k].unsqueeze(-1)
        # for k,v in batch.items():
        #     print(k, v.shape)

        for k, v in batch.items():
            print(k, v.shape)

        for k, v in batch.items():
            if k == "obj_idx" or k == "instance_ids":
                continue
            batch[k] = v.squeeze(0)
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)

        for k, v in batch.items():
            if k == "radii":
                batch[k] = v.squeeze(0)
            print("k,v", k, v.shape)
        W, H = self.hparams.img_wh
        ret = self.render_rays(batch, batch_idx)
        # rank = dist.get_rank()
        rank = 0
        if rank == 0:
            grid_img = visualize_val_rgb((W, H), batch, ret)
            self.logger.experiment.log({"val/GT_pred rgb": wandb.Image(grid_img)})

        # return self.render_rays(batch, batch_idx)

        # return self.render_rays(batch, batch_idx)

    # def test_step(self, batch, batch_idx):
    #     return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "obj_idx" or k == "instance_ids":
                continue
            batch[k] = v.squeeze(0)
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)

        for k, v in batch.items():
            if k == "radii":
                batch[k] = v.squeeze(0)
            print("k,v", k, v.shape)
        return self.render_rays_test(batch, batch_idx)

    def configure_optimizers(self):
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
        step = self.trainer.global_step
        max_steps = 1000000

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=2048,
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
    #     # val_image_sizes = self.trainer.datamodule.val_image_sizes
    #     rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
    #     targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
    #     psnr_mean = self.psnr_each(rgbs, targets).mean()
    #     ssim_mean = self.ssim_each(rgbs, targets).mean()
    #     lpips_mean = self.lpips_each(rgbs, targets).mean()

    #     for (i, rgb) in enumerate(rgbs):
    #         rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
    #         self.logger.experiment.log({
    #             "Val images": [wandb.Image(rgbimg)]
    #         })

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
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)

        psnr = self.psnr(rgbs, targets, None, None, None)
        ssim = self.ssim(rgbs, targets, None, None, None)
        lpips = self.lpips(rgbs, targets, None, None, None)

        all_obj_rgbs, all_target_rgbs = get_obj_rgbs_from_segmap(
            instance_masks, rgbs, targets
        )

        psnr_obj = self.psnr(all_obj_rgbs, all_target_rgbs, None, None, None)
        print("psnr obj", psnr_obj)

        # psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # lpips = self.lpips(
        #     rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        # )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)
        print("psnr, ssim, lpips", psnr, ssim, lpips)
        self.log("test/psnr_obj", psnr_obj["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(
                "ckpts", self.hparams.exp_name, self.hparams.render_name
            )
            os.makedirs(image_dir, exist_ok=True)
            store_image(image_dir, rgbs, "image")

            result_path = os.path.join("ckpts", self.hparams.exp_name, "results.json")
            write_stats(result_path, psnr, ssim, lpips, psnr_obj)

        return psnr, ssim, lpips

    def interlevel_loss(self, ray_history):
        last_ray_results = ray_history[-1]
        c = last_ray_results["sdist"].detach()
        w = last_ray_results["weights"].detach()
        loss_interlevel = 0.0
        for ray_results in ray_history[:-1]:
            cp = ray_results["sdist"]
            wp = ray_results["weights"]
            loss_interlevel += torch.mean(helper.lossfun_outer(c, w, cp, wp))
        return loss_interlevel

    def distortion_loss(self, ray_history):
        last_ray_results = ray_history[-1]
        c = last_ray_results["sdist"]
        w = last_ray_results["weights"]
        loss = torch.mean(helper.lossfun_distortion(c, w))
        return loss
