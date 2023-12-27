import numpy as np

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def read_poses(pose_dir_train, img_files_train):
    pose_file_train = os.path.join(pose_dir_train, 'pose.json')
    with open(pose_file_train, "r") as read_content:
        data = json.load(read_content)

    focal = data['focal']
    fov = data['fov']
    img_wh = data['img_size']
    obj_location = np.array(data["obj_location"])
    all_c2w_train = []

    for img_file in img_files_train:
        c2w = np.array(data['transform'][img_file.split('.')[0]])
        c2w[:3, 3] = c2w[:3, 3] - obj_location
        all_c2w_train.append(convert_pose_PD_to_NeRF(c2w))

    all_c2w_train = np.array(all_c2w_train)
    pose_scale_factor = 1. / np.max(np.abs(all_c2w_train[:, :3, 3]))

    all_c2w_train[:, :3, 3] *= pose_scale_factor

    # new_all_c2w = []
    # for c2w in all_c2w_train:
    #     c2w_convert = convert_pose(c2w)
    #     new_all_c2w.append(c2w_convert)
    # all_c2w_train = np.array(new_all_c2w)

    #We use pose scale factor during training and not for visualization
    # all_c2w_train = all_c2w_train

    bbox_dimensions = []

    all_translations= []
    all_rotations = []

    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimensions.append(np.array(v)* pose_scale_factor)

            #New scene 200 uncomment here
            all_rotations.append(data["obj_rotations"][k])

            translation = np.array(data['obj_translations'][k] - obj_location) * pose_scale_factor
            all_translations.append(translation)

    return all_c2w_train, focal, img_wh, bbox_dimensions, all_rotations, all_translations

import math
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

if __name__ == '__main__':
    import os
    import json
    new = True

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='PDMultiObjv6/train/SF_6thAndMission_medium0')
    #that is the only argument we need for now

    args = parser.parse_args()
    base_dir = args.base_dir

    # base_dir = '/home/zubairirshad/Downloads/PDMultiObjv6/train/SF_6thAndMission_medium0'

    img_files = os.listdir(os.path.join(base_dir, 'train','rgb'))
    img_files.sort()

    all_c2w, focal, img_size, bbox_dimensions, all_rotations, all_translations = read_poses(pose_dir_train = os.path.join(base_dir,'train', 'pose'), img_files_train = img_files)

    print("image size", img_size)
    print("img_files", img_files)
    fov = focal2fov(focal, img_size[0])
    H = 480
    W = 640


    transforms_data = {
        "camera_angle_x": fov,
        "frames": []
    }

    output_file = os.path.join(base_dir, "transforms_train.json")
    for c2w, img_file in zip(all_c2w, img_files):
        frame_data = {
            "file_path": os.path.join("./", "train", "rgb", img_file.split('.')[0]),
            "transform_matrix": np.array(c2w).tolist()
        }
        transforms_data["frames"].append(frame_data)

    with open(output_file, "w") as json_file:
        json.dump(transforms_data, json_file, indent=4)

    print(f"Transforms data saved to {output_file}")


