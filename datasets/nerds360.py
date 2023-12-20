import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
from .ray_utils import *

def read_poses(pose_dir_train, img_files_train, output_boxes=False):
    pose_file_train = os.path.join(pose_dir_train, "pose.json")
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data["focal"]
    img_wh = data["img_size"]
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data["transform"][img_file.split(".")[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    pose_scale_factor = 1.0 / np.max(np.abs(all_c2w_train[:, :3, 3]))
    all_c2w_train[:, :3, 3] *= pose_scale_factor
    all_c2w_val = all_c2w_train[100:]
    all_c2w_train = all_c2w_train[:100]
    # Get bounding boxes for object MLP training only
    if output_boxes:
        all_boxes = []
        all_translations = []
        all_rotations = []
        for k, v in data["bbox_dimensions"].items():
            bbox = np.array(v)
            all_boxes.append(bbox * pose_scale_factor)
            # New scene 200 uncomment here
            all_rotations.append(data["obj_rotations"][k])
            translation = (
                np.array(data["obj_translations"][k]) - obj_location
            ) * pose_scale_factor
            all_translations.append(translation)
        RTs = {"R": all_rotations, "T": all_translations, "s": all_boxes}
        return all_c2w_train, all_c2w_val, focal, img_wh, RTs, pose_scale_factor
    else:
        return all_c2w_train, all_c2w_val, focal, img_wh, pose_scale_factor


def read_poses_val(pose_dir_train, img_files_train, pose_scale_factor):
    pose_file_train = os.path.join(pose_dir_train, "pose.json")
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data["transform"][img_file.split(".")[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    all_c2w_train[:, :3, 3] *= pose_scale_factor

    return all_c2w_train


def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.03
    # radii = 0
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


class NeRDS360(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_wh=(1600, 1600),
        white_back=False,
        model_type="vanilla",
        eval_inference=None,
    ):
        self.root_dir = root_dir
        self.split = split
        print("img_wh", img_wh)
        self.model_type = model_type
        self.img_wh = img_wh
        self.define_transforms()
        self.read_meta()
        self.white_back = False
        self.eval_inference = eval_inference
        w, h = self.img_wh
        if self.eval_inference is not None:
            # eval_num = int(self.eval_inference[0])
            num = 99
            # num = 40
            self.image_sizes = np.array([[h, w] for i in range(num)])
        else:
            self.image_sizes = np.array([[h, w] for i in range(1)])

    def read_meta(self):
        base_dir_train = os.path.join(self.root_dir, "train")
        img_files_train = os.listdir(os.path.join(base_dir_train, "rgb"))
        img_files_train.sort()

        # base_dir_test = os.path.join(self.root_dir, "val")
        # img_files_test = os.listdir(os.path.join(base_dir_test, "rgb"))
        # img_files_test.sort()
        # pose_dir_test = os.path.join(self.root_dir, "val", "pose")

        # base_dir_test = os.path.join(self.root_dir, 'val')
        # img_files_test = os.listdir(os.path.join(base_dir_test, 'rgb'))
        # img_files_test.sort()

        # self.all_unique_ids = [1167, 1168, 1169, 1170]
        self.near = 0.2
        self.far = 3.0
        pose_dir_train = os.path.join(self.root_dir, "train", "pose")
        # pose_dir_val = os.path.join(self.root_dir, 'val', 'pose')

        if self.split == "train":
            all_c2w, all_c2w_val, self.focal, self.img_size, _, _ = read_poses(
                pose_dir_train, img_files_train, output_boxes=True
            )
            # self.img_files_val = img_files_train[100:]
            # self.all_c2w_val = all_c2w_val
            # self.img_files_val = img_files_train[:100]
            # self.all_c2w_val = all_c2w[:100]
            # self.base_dir_val = base_dir_train
        elif self.split == "val":
            all_c2w, all_c2w_val, self.focal, self.img_size, _, _ = read_poses(
                pose_dir_train, img_files_train, output_boxes=True
            )
            # self.img_files_val = img_files_train[100:]
            # self.all_c2w_val = all_c2w_val

            self.img_files_val = img_files_train[100:]
            self.all_c2w_val = all_c2w_val
            self.base_dir_val = base_dir_train
        else:
            (
                all_c2w,
                all_c2w_val,
                self.focal,
                self.img_size,
                _,
                poses_scale_factor,
            ) = read_poses(pose_dir_train, img_files_train, output_boxes=True)

            self.img_files_val = img_files_train[100:]
            self.all_c2w_val = all_c2w_val
            self.base_dir_val = base_dir_train
            # all_c2w = read_poses_val(pose_dir_test, img_files_test, poses_scale_factor)
            # self.img_files_val = img_files_test
            # self.all_c2w_val = all_c2w
            # self.base_dir_val = base_dir_test
            # img_files_train = img_files_test

            # only while generating circular round trajectory
            # ref_pose = all_c2w[0]
            # all_c2w_test = []
            # for i in range(40):
            #     all_c2w_test.append(move_camera_pose(np.copy(ref_pose), i / 40))
            # all_c2w = np.array(all_c2w_test)

        print("all c2w", all_c2w.shape)
        w, h = self.img_wh
        print("self.focal", self.focal)
        self.focal *= (
            self.img_wh[0] / self.img_size[0]
        )  # modify focal length to match size self.img_wh
        print("self.focal after", self.focal)

        if (
            self.split == "train" or self.split == "test"
        ):  # create buffer of all rays and rgb data
            self.all_rays = []
            self.all_rgbs = []
            self.all_rays_d = []
            self.all_radii = []

            # num = 3
            # src_views_num = [0, 38, 44, 94, 48]
            # if num ==3:
            # src_views_num = [0, 38, 44]
            # src_views_num = [0]
            # elif num ==5:
            #     src_views_num = [7, 28, 50, 66, 75]
            # elif num ==7:
            #     src_views_num = [7, 28, 39, 50, 64, 66, 75]
            # elif num ==9:
            #     src_views_num = [7, 21, 28, 39, 45, 50, 64, 66, 75]
            # elif num ==1:
            #     src_views_num = [7]
            NV = 100

            for train_image_id in range(0, NV):
                # for i, img_name in enumerate(self.img_files):

                # train for all views i.e. overfitting scenario
                # if train_image_id not in src_views_num:
                #     continue
                img_name = img_files_train[train_image_id]
                c2w = all_c2w[train_image_id]
                directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)
                c2w = torch.FloatTensor(c2w)[:3, :4]
                rays_o, view_dirs, rays_d, radii = get_rays(
                    directions, c2w, output_view_dirs=True, output_radii=True
                )

                img = Image.open(os.path.join(base_dir_train, "rgb", img_name))
                img = img.resize((w, h), Image.LANCZOS)
                img = self.transform(img)  # (h, w, 3)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
                self.all_rays += [
                    torch.cat(
                        [
                            rays_o,
                            view_dirs,
                            self.near * torch.ones_like(rays_o[:, :1]),
                            self.far * torch.ones_like(rays_o[:, :1]),
                        ],
                        1,
                    )
                ]  # (h*w, 8)
                self.all_rays_d += [view_dirs]
                self.all_radii += [radii.unsqueeze(-1)]
                self.all_rgbs += [img]

            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.all_rays_d = torch.cat(self.all_rays_d, 0)
            self.all_radii = torch.cat(self.all_radii, 0)

            print("self.all rays", self.all_rays.shape)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == "train" or self.split == "test":
            return len(self.all_rays)
        if self.split == "val" or self.split == "test_val":
            if self.eval_inference is not None:
                # num = int(self.eval_inference[0])
                return 99
                # return 40
            else:
                return 1
        return len(self.meta)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "test":  # use data in the buffers
            # rand_instance_id = torch.randint(0, len(self.all_unique_ids), (1,))
            if self.model_type == "vanilla":
                sample = {
                    "rays": self.all_rays[idx],
                    "rgbs": self.all_rgbs[idx],
                }
            else:
                sample = {}
                sample["rays_o"] = self.all_rays[idx][:3]
                sample["rays_d"] = self.all_rays_d[idx]
                sample["viewdirs"] = self.all_rays[idx][3:6]
                sample["radii"] = self.all_radii[idx]
                sample["target"] = self.all_rgbs[idx]
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])

        elif (
            self.split == "val" or self.split == "test_val"
        ):  # create data for each image separately
            if self.eval_inference:
                # num = int(self.eval_inference[0])
                # if num ==3:
                #     src_views_num = [7, 50, 66]
                # elif num ==5:
                #     src_views_num = [7, 28, 50, 66, 75]
                # elif num ==7:
                #     src_views_num = [7, 28, 39, 50, 64, 66, 75]
                # elif num ==9:
                #     src_views_num = [7, 21, 28, 39, 45, 50, 64, 66, 75]
                # elif num ==1:
                #     src_views_num = [7]

                # all_num = list(range(0,100))
                # eval_list = list(set(all_num).difference(src_views_num))
                # dest_view_num = eval_list[idx]
                # val_idx = dest_view_num
                val_idx = idx
            else:
                val_idx = 0

            img_name = self.img_files_val[val_idx]
            w, h = self.img_wh
            c2w = self.all_c2w_val[val_idx]
            directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            rays_o, view_dirs, rays_d, radii = get_rays(
                directions, c2w, output_view_dirs=True, output_radii=True
            )
            img = Image.open(os.path.join(self.base_dir_val, "rgb", img_name))
            img = img.resize((w, h), Image.LANCZOS)
            img = self.transform(img)  # (h, w, 3)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA

            seg_mask = Image.open(
                os.path.join(self.base_dir_val, "semantic_segmentation_2d", img_name)
            )
            # seg_mask = seg_mask.resize((w,h), Image.LANCZOS)
            seg_mask = np.array(seg_mask)
            seg_mask[seg_mask != 5] = 0
            seg_mask[seg_mask == 5] = 1
            seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            instance_mask = seg_mask > 0
            instance_mask = self.transform(instance_mask).view(-1)

            rays = torch.cat(
                [
                    rays_o,
                    view_dirs,
                    self.near * torch.ones_like(rays_o[:, :1]),
                    self.far * torch.ones_like(rays_o[:, :1]),
                ],
                1,
            )  # (h*w, 8)
            if self.model_type == "vanilla":
                sample = {
                    "rays": rays,
                    "rgbs": img,
                    "img_wh": self.img_wh,
                }
            else:
                sample = {}
                sample["rays_o"] = rays[:, :3]
                sample["rays_d"] = view_dirs
                sample["viewdirs"] = rays[:, 3:6]
                sample["target"] = img
                sample["instance_mask"] = instance_mask
                sample["radii"] = radii
                sample["multloss"] = np.zeros((sample["rays_o"].shape[0], 1))
                sample["normals"] = np.zeros_like(sample["rays_o"])
        return sample
