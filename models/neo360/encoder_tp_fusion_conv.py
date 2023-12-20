import torch
from torch import nn
import torch.nn.functional as F
from models.neo360.util import *
import numpy as np
import torch.autograd.profiler as profiler
from models.neo360.encoder_pn import SpatialEncoder
import models.neo360.helper as helper
from torch import linalg as LA
from torchvision import transforms as T

def contract_samples(x, order=float("inf")):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag)), mag


def inverse_contract_samples(x, mag_origial, order=float("inf")):
    mag = LA.norm(x, order, dim=-1)[..., None]
    return torch.where(mag < 1, x, (x * mag_origial) / (2 - (1 / mag_origial)))


def unprocess_images(normalized_images, encoder_type="resnet"):
    if encoder_type == "resnet":
        inverse_transform = T.Compose(
            [
                T.Normalize(
                    (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5), (1 / 0.5, 1 / 0.5, 1 / 0.5)
                )
            ]
        )
    else:
        inverse_transform = T.Compose(
            [
                T.Normalize(
                    (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                    (1 / 0.229, 1 / 0.224, 1 / 0.225),
                )
            ]
        )
    return inverse_transform(normalized_images)


# def get_c(samples, imgs, poses, focal, c, encoder_type = 'resnet', attn = False, viewdirs = None, use_same_stride = False):
#     focal = focal[0].unsqueeze(-1).repeat((1, 2))
#     focal[..., 1] *= -1.0
#     c = c[0].unsqueeze(0)

#     B, N_samples, _ = samples.shape

#     # imgs_unprocess = unprocess_images(imgs, encoder_type = encoder_type)
#     imgs_unprocess = unprocess_images(imgs, encoder_type = encoder_type)

#     if use_same_stride:
#         #since feature vector was of the same size
#         imgs_unprocess = F.interpolate(imgs_unprocess, size=(120,160), mode='bilinear', align_corners=False)
#         focal = focal/2
#         c = c/2
#         NV, C, height, width = imgs_unprocess.shape
#     else:
#         NV, C, height, width = imgs_unprocess.shape

#     samples = samples[:,:,:3].reshape(-1,3).unsqueeze(0).float()
#     cam_xyz = world2camera(samples, poses, NV)
#     uv = projection(cam_xyz, focal, c)

#     im_x = uv[:,:, 0]
#     im_y = uv[:,:, 1]
#     im_grid = torch.stack([2 * im_x / (width - 1) - 1, 2 * im_y / (height - 1) - 1], dim=-1)
#     mask_z = cam_xyz[:,:,2]<1e-3
#     mask = im_grid.abs() <= 1
#     mask = (mask.sum(dim=-1) == 2) & (mask_z)
#     im_grid = im_grid.unsqueeze(2)
#     colors = []
#     for i, idx in enumerate(range(imgs_unprocess.shape[0])):
#         imgs_feat = F.grid_sample(imgs_unprocess[idx, :, :, :].unsqueeze(0), im_grid[idx, :, :].unsqueeze(0), align_corners=True, mode='bilinear', padding_mode='zeros')
#         # imgs_feat[mask.unsuqeeze(0).unsqueeze(-1)==False] = 0
#         colors.append(imgs_feat.squeeze(-1).permute(0,1,2))
#         # colors[...,i*C:i*C+C] = imgs_feat[0].squeeze(-1).permute(1,0)
#     colors = torch.cat(colors, dim=0)
#     if attn:
#         # colors = torch.cat((colors,cam_xyz.permute(0, 2, 1)) , dim=0)
#         min_deg_point: int = 0
#         max_deg_point: int = 10
#         deg_point_view = 4
#         samples_enc = helper.pos_enc(
#             cam_xyz,
#             min_deg_point,
#             max_deg_point,
#         )
#         viewdirs = world2camera_viewdirs(viewdirs.unsqueeze(0), poses, NV)
#         viewdirs_enc = helper.pos_enc(viewdirs, 0, deg_point_view)

#         viewdirs_enc = torch.tile(viewdirs_enc[:, None, :], (1, N_samples, 1)).reshape(
#                 NV, -1, viewdirs_enc.shape[-1]
#             )

#         colors = torch.cat((colors.permute(0,2,1), samples_enc, viewdirs_enc) , dim=-1)

#     else:
#         min_deg_point: int = 0
#         max_deg_point: int = 10
#         deg_point_view = 4
#         samples_enc = helper.pos_enc(
#             cam_xyz,
#             min_deg_point,
#             max_deg_point,
#         )
#         viewdirs = world2camera_viewdirs(viewdirs.unsqueeze(0), poses, NV)
#         viewdirs_enc = helper.pos_enc(viewdirs, 0, deg_point_view)

#         viewdirs_enc = torch.tile(viewdirs_enc[:, None, :], (1, N_samples, 1)).reshape(
#                 NV, -1, viewdirs_enc.shape[-1]
#             )
#         colors = colors.permute(0,2,1)
#         colors[mask.unsqueeze(-1).repeat(1,1,colors.shape[-1])==False] = 0
#         viewdirs_enc[mask.unsqueeze(-1).repeat(1,1,viewdirs_enc.shape[-1])==False] = 0
#         colors = torch.cat((colors, samples_enc, viewdirs_enc) , dim=-1)

#     return colors


def index_grid(
    samples, scene_grid_xz, scene_grid_xy, scene_grid_yz, poses, src_views_num
):
    samples = samples.reshape(-1, 3).unsqueeze(0)
    samples_cam = world2camera(samples, poses, src_views_num)

    uv_xz = samples_cam[:, :, [0, 2]].float()
    uv_yz = samples_cam[:, :, [1, 2]].float()
    uv_xy = samples_cam[:, :, [0, 1]].float()
    scale_factor = 1

    uv_xz = (uv_xz / scale_factor).unsqueeze(2)  # (B, N, 1, 2)
    uv_yz = (uv_yz / scale_factor).unsqueeze(2)  # (B, N, 1, 2)
    uv_xy = (uv_xy / scale_factor).unsqueeze(2)  # (B, N, 1, 2)
    
    # print("=====================Triplane indexing\n\n\n")
    # print("scene_grid_xz", scene_grid_xz.shape)
    # print("uv_xz", uv_xz.shape)
    # print("=====================Triplane indexing\n\n\n")


    #batched indexing
    # chunk = 50000
    # scene_latent_xz = torch.zeros((scene_grid_xz.shape[0],scene_grid_xz.shape[1],uv_xz.shape[1], 1), device=scene_grid_xz.device, dtype=torch.float, requires_grad=scene_grid_xz.requires_grad)
    # scene_latent_xy = torch.zeros((scene_grid_xz.shape[0],scene_grid_xz.shape[1],uv_xz.shape[1], 1), device=scene_grid_xz.device, dtype=torch.float, requires_grad=scene_grid_xz.requires_grad)
    # scene_latent_yz = torch.zeros((scene_grid_xz.shape[0],scene_grid_xz.shape[1],uv_xz.shape[1], 1), device=scene_grid_xz.device, dtype=torch.float, requires_grad=scene_grid_xz.requires_grad)
    
    # for i in range(0, uv_xz.shape[1], chunk):
    #     chunk_samples_xz = F.grid_sample(
    #         scene_grid_xz,
    #         uv_xz[:, i:i + chunk, :, :],
    #         align_corners=True,
    #         mode="bilinear",
    #         padding_mode="zeros"
    #             )
    #     detached_samples = chunk_samples_xz.detach()
    #     scene_latent_xz[:,:,i:i + chunk,:] = detached_samples
        
    #     chunk_samples_yz = F.grid_sample(
    #         scene_grid_yz,
    #         uv_yz[:, i:i + chunk, :, :],
    #         align_corners=True,
    #         mode="bilinear",
    #         padding_mode="zeros"
    #             )
    #     detached_samples = chunk_samples_yz.detach()
    #     scene_latent_yz[:,:,i:i + chunk,:] = detached_samples

    #     chunk_samples_xy = F.grid_sample(
    #         scene_grid_xy,
    #         uv_xy[:, i:i + chunk, :, :],
    #         align_corners=True,
    #         mode="bilinear",
    #         padding_mode="zeros"
    #             )
    #     detached_samples = chunk_samples_xy.detach()
    #     scene_latent_xy[:,:,i:i + chunk,:] = detached_samples

    scene_latent_xz = F.grid_sample(
        scene_grid_xz,
        uv_xz,
        align_corners=True,
        mode="bilinear",  # "nearest",
        padding_mode="zeros",
    )

    scene_latent_xy = F.grid_sample(
        scene_grid_xy,
        uv_xy,
        align_corners=True,
        mode="bilinear",  # "nearest",
        padding_mode="zeros",
    )

    scene_latent_yz = F.grid_sample(
        scene_grid_yz,
        uv_yz,
        align_corners=True,
        mode="bilinear",  # "nearest",
        padding_mode="zeros",
    )

    output = torch.sum(
        torch.stack([scene_latent_xz, scene_latent_xy, scene_latent_yz]), dim=0
    )
    del samples
    output = output[..., 0].permute(0, 2, 1).reshape(-1, 128)
    return output


# def index_grid(samples, scene_grid_xz, scene_grid_xy, scene_grid_yz):
#     """
#     Get pixel-aligned image features at 2D image coordinates
#     :param uv (B, N, 2) image points (x,y)
#     :param cam_z ignored (for compatibility)
#     :param image_size image size, either (width, height) or single int.
#     if not specified, assumes coords are in [-1, 1]
#     :param z_bounds ignored (for compatibility)
#     :return (B, L, N) L is latent size
#     """
#     scale_factor = 1

#     uv_xz = samples[:, :, [0, 2]].reshape(-1,2).unsqueeze(0).float()
#     uv_yz = samples[:, :, [1, 2]].reshape(-1,2).unsqueeze(0).float()
#     uv_xy = samples[:, :, [0, 1]].reshape(-1,2).unsqueeze(0).float()

#     uv_xz = (uv_xz/scale_factor).unsqueeze(2)  # (B, N, 1, 2)
#     uv_yz = (uv_yz/scale_factor).unsqueeze(2)  # (B, N, 1, 2)
#     uv_xy = (uv_xy/scale_factor).unsqueeze(2)  # (B, N, 1, 2)

#     scene_latent_xz = F.grid_sample(scene_grid_xz,
#                                 uv_xz,
#                                 align_corners=True,
#                                 mode="bilinear",  # "nearest",
#                                 padding_mode="zeros", )

#     scene_latent_xy = F.grid_sample(scene_grid_xy,
#                                 uv_xy,
#                                 align_corners=True,
#                                 mode="bilinear",  # "nearest",
#                                 padding_mode="zeros", )

#     scene_latent_yz = F.grid_sample(scene_grid_yz,
#                                 uv_yz,
#                                 align_corners=True,
#                                 mode="bilinear",  # "nearest",
#                                 padding_mode="zeros", )

#     output = torch.sum(torch.stack([scene_latent_xz, scene_latent_xy, scene_latent_yz]), dim=0)
#     del samples
#     return output[..., 0]


def init_weights_kaiming(m):
    if type(m) == nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight)
        if hasattr(m, "bias"):
            nn.init.uniform_(m.bias, -1e-3, 1e-3)


class DepthPillarEncoder(nn.Module):
    def __init__(self, inp_ch, LS):
        super().__init__()
        self.common_branch = nn.Sequential(
            nn.Linear(inp_ch, LS),
            nn.ReLU(inplace=True),
            nn.Linear(LS, LS),
            nn.ReLU(inplace=True),
        )
        self.depth_encoder = nn.Linear(LS, LS)
        self.common_branch.apply(init_weights_kaiming)
        self.depth_encoder.apply(init_weights_kaiming)

    def forward(self, x):
        out = self.common_branch(x)
        out = self.depth_encoder(out)
        return out


class GridEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        encoder_type="resnet",
        contract=False,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="zeros",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        # grid_size=[64, 64, 64],
        grid_size=[64, 64, 64],
        xyz_min=None,
        xyz_max=None,
        use_transformer=False,
        use_stride=False,
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflzerosection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()
        self.contract = contract
        self.grid_size = grid_size
        self.encoder_type = encoder_type
        self.use_transformer = use_transformer
        self.use_stride = use_stride

        self.side_lengths = [1, 1, 1]
        # if encoder_type == 'resnet':
        self.spatial_encoder = SpatialEncoder(
            backbone="resnet34",
            pretrained=True,
            num_layers=4,
            index_interp="bilinear",
            index_padding="zeros",
            # index_padding="border",
            upsample_interp="bilinear",
            feature_scale=1.0,
            use_first_pool=True,
            norm_type="batch",
        )
        # else:
        #     self.spatial_encoder = ResUNet()

        self.latent_size = (
            self.spatial_encoder.latent_size
        )  # self.spatial_encoder.latent_size/8
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        LS = self.latent_size
        # To encode latent from spatial encoder with camera depth
        self.depth_fc = DepthPillarEncoder(
            inp_ch=self.spatial_encoder.latent_size + 3 + 3, LS=LS
        )

        # ###====================================================\n\n\n
        # ###TRANSFORMER BLOC FOR MVFUSION----------------------\n\n\n
        # if self.use_transformer:
        #     encoder_layer = nn.TransformerEncoderLayer(128, 2, 256)
        #     self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 2)
        # ###====================================================\n\n\n
        self.pillar_aggregator_xz = nn.Sequential(
            nn.Linear(LS + 1, LS), nn.ReLU(inplace=True), nn.Linear(LS, 1)
        )

        self.pillar_aggregator_yz = nn.Sequential(
            nn.Linear(LS + 1, LS), nn.ReLU(inplace=True), nn.Linear(LS, 1)
        )
        self.pillar_aggregator_xy = nn.Sequential(
            nn.Linear(LS + 1, LS), nn.ReLU(inplace=True), nn.Linear(LS, 1)
        )

        self.floorplan_convnet_xy = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(120, 160), mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )

        self.floorplan_convnet_yz = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(120, 160), mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )

        self.floorplan_convnet_xz = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(120, 160), mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
        )

        self.floorplan_convnet_xy.apply(init_weights_kaiming)
        self.floorplan_convnet_yz.apply(init_weights_kaiming)
        self.floorplan_convnet_xz.apply(init_weights_kaiming)

        self.pillar_aggregator_xz.apply(init_weights_kaiming)
        self.pillar_aggregator_yz.apply(init_weights_kaiming)
        self.pillar_aggregator_xy.apply(init_weights_kaiming)

    def get_resnet_feats(self, cam_xyz, focal, c, W, H):
        # mask_z = cam_xyz[:,:,2]<1e-3

        # focal = focal/2
        # c = c/2
        # W = W/2
        # H = H/2

        uv = projection(cam_xyz, focal, c)
        latent, _ = self.spatial_encoder.index(
            uv, None, torch.tensor([W, H], device="cuda")
        )  # (SB * NS, latent, B)

        # mask = (mask.sum(dim=-1) == 2) & (mask_z)
        # latent[mask.unsqueeze(1).repeat(1, latent.shape[1], 1) == False] = 0
        return latent

    def forward(self, images, poses, focal, c):
        """
        For extracting ResNet's features.
        :param images (SB, NV, C, H, W)
        :param poses (SB*NV, 4, 4)
        :param focal focal length (SB) or (SB, 2)
        :param c principal point (SB) or (SB, 2)
        :return latent (SB, latent_size, H, W)
        """
        world_grid = get_world_grid(
            [
                [-self.side_lengths[0], self.side_lengths[0]],
                [-self.side_lengths[1], self.side_lengths[1]],
                [0, self.side_lengths[2]],
            ],
            [int(self.grid_size[0]), int(self.grid_size[1]), int(self.grid_size[2])],
            device=images.device,
        )  # (1, grid_size**3, 3)

        focal = focal[0].unsqueeze(-1).repeat((1, 2))
        focal[..., 1] *= -1.0
        c = c[0].unsqueeze(0)

        # the resnet stride to downscale the intrinsics by

        NV, C, H, W = images.shape

        # _, _, H, W = images.shape
        self.spatial_encoder(images)

        # world_grid = inverse_contract_samples(self.world_grid.clone(), self.mag_original.clone())
        # world_grid = world_grid.unsqueeze(0)

        # world_grids = repeat_interleave(self.world_grid.clone(),
        #                                         NV).cuda()  # (SB*NV, NC, 3) NC: number of grid cells

        world_grids = repeat_interleave(
            world_grid.clone(), NV
        )  # (SB*NV, NC, 3) NC: number of grid cells
        camera_grids = world2camera(world_grids, poses)

        masks = camera_grids[:, :, 2] < 1e-3

        camera_pts_dir = world_grids - poses[:, None, :3, -1]
        camera_pts_dir_norm = torch.norm(camera_pts_dir + 1e-9, dim=-1)
        # print("camera_pts_dir", camera_pts_dir.shape)
        # print("camera_pts_dir_norm", camera_pts_dir_norm)
        # print("camera_pts_dir_norm", camera_pts_dir_norm.shape)
        camera_pts_dir = camera_pts_dir / camera_pts_dir_norm[:, :, None]
        camera_pts_dir = camera_pts_dir * masks[:, :, None]

        # # Projecting points in camera coordinates on the image plane
        # uv = projection(camera_grids, focal, c)  # [f, -f]

        # latent = self.get_resnet_feats(camera_grids, features, focal, c, W_feat, H_feat, stride = stride)

        latent = self.get_resnet_feats(camera_grids, focal, c, W, H)
        _, L, _ = latent.shape  # (NV, L, grid_size**3)
        # latent[mask_expand == False] = 0
        latent = torch.cat(
            [latent, camera_grids.permute(0, -1, 1), camera_pts_dir.permute(0, -1, 1)],
            1,
        )
        latent = latent.reshape(
            NV, L + 3 + 3, self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        ).permute(
            0, 2, 1
        )  # Input to the linear layer # (SB*T*NV, grid_size**3, L+1)
        latent = self.depth_fc(latent)
        latent = latent.reshape(
            1, NV, self.grid_size[0], self.grid_size[1], self.grid_size[2], L
        )

        latent = latent.squeeze(0)

        world_grid_x = world_grids.reshape(
            1, NV, self.grid_size[0], self.grid_size[1], self.grid_size[2], 3
        )[:, 0, ..., 0:1]
        world_grid_x = world_grid_x.repeat(latent.shape[0], 1, 1, 1, 1)

        world_grid_y = world_grids.reshape(
            1, NV, self.grid_size[0], self.grid_size[1], self.grid_size[2], 3
        )[:, 0, ..., 1:2]
        world_grid_y = world_grid_y.repeat(latent.shape[0], 1, 1, 1, 1)

        world_grid_z = world_grids.reshape(
            1, NV, self.grid_size[0], self.grid_size[1], self.grid_size[2], 3
        )[:, 0, ..., 2:3]
        world_grid_z = world_grid_z.repeat(latent.shape[0], 1, 1, 1, 1)

        latent_inp_x = torch.cat([latent, world_grid_x], -1)
        latent_inp_y = torch.cat([latent, world_grid_y], -1)
        latent_inp_z = torch.cat([latent, world_grid_z], -1)

        weights_yz = torch.softmax(
            self.pillar_aggregator_yz(latent_inp_x), dim=1
        )  # (SB, T, X, Z, Y, 1)
        weights_xz = torch.softmax(
            self.pillar_aggregator_xz(latent_inp_y), dim=2
        )  # (SB, T, X, Z, Y, 1)
        weights_xy = torch.softmax(
            self.pillar_aggregator_xy(latent_inp_z), dim=3
        )  # (SB, T, X, Z, Y, 1)

        floorplans_yz = (latent * weights_yz).sum(1)  # (SB, T, X, Z, L)
        floorplans_xz = (latent * weights_xz).sum(2)  # (SB, T, X, Z, L)
        floorplans_xy = (latent * weights_xy).sum(3)  # (SB, T, X, Z, L)

        del world_grids, camera_grids

        grid_yz = floorplans_yz.permute(
            0, -1, 1, 2
        )  # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        scene_grid_yz = self.floorplan_convnet_yz(grid_yz)  # (SB*T, L, X, Z)

        grid_xz = floorplans_xz.permute(
            0, -1, 1, 2
        )  # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        scene_grid_xz = self.floorplan_convnet_xz(grid_xz)  # (SB*T, L, X, Z)

        grid_xy = floorplans_xy.permute(
            0, -1, 1, 2
        )  # .reshape(SB, T, L, int(self.grid_size[0]/sfactor), int(self.grid_size[2]/sfactor))
        scene_grid_xy = self.floorplan_convnet_xy(grid_xy)  # (SB*T, L, X, Z)

        return scene_grid_xz, scene_grid_xy, scene_grid_yz
        # return 0
