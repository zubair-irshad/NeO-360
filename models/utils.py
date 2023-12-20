import os

import imageio
import numpy as np
from PIL import Image
import json
import cv2
from numpy import savez_compressed
from torchvision.ops import masks_to_boxes, box_iou
import torch

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def norm8b(x):
    x = (x - x.min()) / (x.max() - x.min())
    return to8b(x)


def store_image(dirpath, rgbs, name):
    for (i, rgb) in enumerate(rgbs):
        imgname = f"{str(i).zfill(3)}.jpg"
        imgname = name+imgname
        rgbimg = Image.fromarray(to8b(rgb.detach().cpu().numpy()))
        imgpath = os.path.join(dirpath, imgname)
        rgbimg.save(imgpath)

def store_depth_img(dirpath, depths, name):
    depth_maps = []
    for (i, depth) in enumerate(depths):
        depth_maps += [depth.detach().cpu().numpy()]

    min_depth = np.min(depth_maps)
    max_depth = np.max(depth_maps)
    depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
    depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]

    for (i, depth) in enumerate(depth_imgs_):
        depthname = f"{str(i).zfill(3)}.jpg"
        depthname = name+depthname
        depth_img = Image.fromarray(depth)
        depthpath = os.path.join(dirpath, depthname)
        depth_img.save(depthpath)

def store_depth_raw(dirpath, depths, name):
    depth_maps = []
    for (i, depth) in enumerate(depths):
        depthname = f"{str(i).zfill(3)}"
        depthname = name+depthname
        depthpath = os.path.join(dirpath, depthname)
        savez_compressed(depthpath, depth.detach().cpu().numpy())


def store_video(dirpath, rgbs, depths):
    rgbimgs = [to8b(rgb.cpu().detach().numpy()) for rgb in rgbs]
    video_dir = os.path.join(dirpath, "videos")
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimwrite(os.path.join(video_dir, "images.mp4"), rgbimgs, fps=20, quality=8)


def write_stats(fpath, *stats):

    d = {}
    for stat in stats:
        d[stat["name"]] = {
            k: float(w)
            for (k, w) in stat.items()
            if k != "name" and k != "scene_wise"
        }

    with open(fpath, "w") as fp:
        json.dump(d, fp, indent=4, sort_keys=True)

def get_boxes_from_segmap(segmap):
    mask_list = []
    ids_list = []
    for i in segmap.unique():
        if i != 0:
            mask = torch.zeros_like(segmap)
            mask[segmap == i] = 1
            mask_list.append(mask[None])
            ids_list.append(i.item())
    masks = torch.cat(mask_list, dim=0)
    boxes = masks_to_boxes(masks)
    return boxes

# def get_obj_rgbs_from_instmask(all_segmap, all_pred_img, all_pred_target):
#     all_obj_rgbs = []
#     all_target_rgbs = []
#     for inst_seg_map, pred, target in zip(all_segmap, all_pred_img, all_pred_target):
#         boxes = get_boxes_from_segmap(inst_seg_map)
#         for box in boxes:
#             xmin, ymin, xmax, ymax = box
#             obj_rgb_pred = pred[int(ymin):int(ymax), int(xmin):int(xmax)]
#             obj_rgb_target = target[int(ymin):int(ymax), int(xmin):int(xmax)]
#             all_obj_rgbs.append(obj_rgb_pred)
#             all_target_rgbs.append(obj_rgb_target)
#     return all_obj_rgbs, all_target_rgbs


def get_obj_rgbs_from_segmap(all_segmap, all_pred_img, all_pred_target):
    all_obj_rgbs = []
    all_target_rgbs = []
    for seg_map, pred, target in zip(all_segmap, all_pred_img, all_pred_target):
            mask = seg_map.unsqueeze(-1).repeat(1,1,3)
            all_obj_rgbs.append(pred[mask])
            all_target_rgbs.append(target[mask])
    return all_obj_rgbs, all_target_rgbs