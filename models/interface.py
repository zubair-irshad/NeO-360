# ------------------------------------------------------------------------------------
# NeO-360
# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# ------------------------------------------------------------------------------------
import json
import os

import numpy as np
import pytorch_lightning as pl
import torch
from piqa.lpips import LPIPS
from piqa.ssim import SSIM

from models.utils import store_image

reshape_2d = lambda x: x.reshape((x.shape[0], -1))
clip_0_1 = lambda x: torch.clip(x, 0, 1).detach()


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = True


    # Utils to reorganize output values from evaluation steps,
    # i.e., validation and test step.
    def alter_gather_cat(self, outputs, key, image_sizes):
        each = torch.cat([output[key] for output in outputs])
        print("each", each.shape)
        all = self.all_gather(each).detach()
        print("all", all.shape)
        if all.dim() == 3:
            all = all.permute((1, 0, 2)).flatten(0, 1)
        if all.shape[-1] ==1:
            all = all.squeeze(-1)
        ret, curr = [], 0
        for (h, w) in image_sizes:
            if all.shape[-1] == 3:
                print("curr, h, w", curr, h, w)
                print("all[curr : curr + h * w]", all[curr : curr + h * w].shape)
                if all[curr : curr + h * w].shape[0] ==0:
                    continue
                ret.append(all[curr : curr + h * w].reshape(h, w, 3))
            else:
                ret.append(all[curr : curr + h * w].reshape(h, w))
            curr += h * w
        return ret

    @torch.no_grad()
    def psnr_each(self, preds, gts):
        psnr_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred, 0, 1)
            gt = torch.clip(gt, 0, 1)
            mse = torch.mean((pred - gt) ** 2)
            psnr = -10.0 * torch.log(mse) / np.log(10)
            psnr_list.append(psnr)
        return torch.stack(psnr_list)

    def mse(self, image_pred, image_gt, valid_mask=None, reduction='mean'):
        value = (image_pred-image_gt)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return torch.mean(value)
        return value

    @torch.no_grad()
    def psnr_legacy(self, image_pred, image_gt, valid_mask=None, reduction='mean'):
        return -10*torch.log10(self.mse(image_pred, image_gt, valid_mask, reduction))

    @torch.no_grad()
    def lpips_legacy(self, pred, gt):
        pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
        gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
        lpips = lpips_model(pred, gt)
        del lpips_model
        return lpips

    @torch.no_grad()
    def ssim_legacy(self, pred, gt):
        ssim_model = SSIM().to(device=self.device)
        pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
        gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
        ssim = ssim_model(pred, gt)
        del ssim_model
        return ssim

    @torch.no_grad()
    def depth_mae_rmse(self, pred, gt):
        abs_diff = (pred - gt).abs()
        mse = float((torch.pow(abs_diff, 2)).mean())
        rmse = torch.sqrt(mse)
        mae = float(abs_diff.mean())
        return rmse, mae

    @torch.no_grad()
    def ssim_each(self, preds, gts):
        ssim_model = SSIM().to(device=self.device)
        ssim_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            ssim = ssim_model(pred, gt)
            ssim_list.append(ssim)
        del ssim_model
        return torch.stack(ssim_list)

    @torch.no_grad()
    def lpips_each(self, preds, gts):
        lpips_model = LPIPS(network="vgg").to(device=self.device)
        lpips_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            lpips = lpips_model(pred, gt)
            lpips_list.append(lpips)
        del lpips_model
        return torch.stack(lpips_list)

    @torch.no_grad()
    def psnr(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "PSNR"
        psnr_list = self.psnr_each(preds, gts)
        print("psnr_list", psnr_list)
        ret["mean"] = psnr_list.mean().item()
        ret["test"] = psnr_list.mean().item()
        # if self.trainer.datamodule.eval_test_only:
        #     ret["test"] = psnr_list.mean().item()
        # else:
        #     ret["train"] = psnr_list[i_train].mean().item()
        #     ret["val"] = psnr_list[i_val].mean().item()
        #     ret["test"] = psnr_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def ssim(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "SSIM"
        ssim_list = self.ssim_each(preds, gts)
        print("ssim_list", ssim_list)
        ret["mean"] = ssim_list.mean().item()
        ret["test"] = ssim_list.mean().item()
        # if self.trainer.datamodule.eval_test_only:
        #     ret["test"] = ssim_list.mean().item()
        # else:
        #     ret["train"] = ssim_list[i_train].mean().item()
        #     ret["val"] = ssim_list[i_val].mean().item()
        #     ret["test"] = ssim_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def lpips(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "LPIPS"
        lpips_list = self.lpips_each(preds, gts)
        ret["mean"] = lpips_list.mean().item()
        ret["test"] = lpips_list.mean().item()
        # if self.trainer.datamodule.eval_test_only:
        #     ret["test"] = lpips_list.mean().item()
        # else:
        #     ret["train"] = lpips_list[i_train].mean().item()
        #     ret["val"] = lpips_list[i_val].mean().item()
        #     ret["test"] = lpips_list[i_test].mean().item()

        return ret

    def write_stats(self, fpath, *stats):

        d = {}
        for stat in stats:
            d[stat["name"]] = {
                k: float(w)
                for (k, w) in stat.items()
                if k != "name" and k != "scene_wise"
            }

        with open(fpath, "w") as fp:
            json.dump(d, fp, indent=4, sort_keys=True)

    def predict_step(self, *args, **kwargs):
        return self.test_step(*args, **kwargs)

    def on_predict_epoch_end(self, outputs):
        # dmodule = self.trainer.datamodule
        image_sizes = self.val_dataset.image_sizes
        # image_sizes = dmodule.image_sizes
        # image_num = len(dmodule.render_poses)
        image_num = len(self.val_dataset)
        all_image_sizes = np.stack([image_sizes[0] for _ in range(image_num)])
        rgbs = self.alter_gather_cat(outputs[0], "rgb", all_image_sizes)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_video")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)
            store_image.store_video(image_dir, rgbs, None)

        return None