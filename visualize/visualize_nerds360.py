import open3d as o3d
import json
import numpy as np
import json

import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.visualizer as pv

from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
import math

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)
    print("directions", directions.shape)
    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def look_at(cam_location, point):
    # Cam points in positive z direction
    forward = point - cam_location
    forward = normalize(forward)

    tmp = np.array([0., -1., 0.])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat, forward

def get_camera_frustum(img_size, focal, C2W, frustum_length=1, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)
    # print("hfov", hfov, vfov)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    #print("frustum_points afters", frustum_points)
    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def convert_pose_spiral(C2W):
    convert_mat = np.zeros((4,4))
    convert_mat[0,1] = 1
    convert_mat[1, 0] = 1
    convert_mat[2, 2] = -1
    convert_mat[3,3] = 1
    C2W = np.matmul(C2W, convert_mat)
    return C2W

def plot_rays(c2w, focal, W, H, fig):
    focal = focal
    directions = get_ray_directions(H, W, focal) # (h, w, 3)
    c2w = torch.FloatTensor(c2w)[:3, :4]
    rays_o, rays_d = get_rays(directions, c2w)

    rays_o = rays_o.numpy()
    rays_d = rays_d.numpy()
    ids = np.random.choice(rays_o.shape[0], int(rays_o.shape[0]*0.5))

    rays_o = rays_o[ids, :]
    rays_d = rays_d[ids, :]
    
    for j in range(2500):
        start = rays_o[j,:]
        end = rays_o[j,:] + rays_d[j,:]*0.02
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(1.0, 0.5, 0.0))

        start = rays_o[j,:] + rays_d[j,:]*0.02
        end = rays_o[j,:] + rays_d[j,:]*2
        line = np.concatenate((start[None, :],end[None, :]), axis=0)
        fig.plot(line, c=(0.0, 1.0, 0.0))

def convert_pose_PD_to_NeRF(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, flip_axes)
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
    #We use pose scale factor during training and not for visualization
    all_c2w_train = all_c2w_train

    bbox_dimensions = []

    all_translations= []
    all_rotations = []

    for k,v in data['bbox_dimensions'].items():
            bbox = np.array(v)
            bbox_dimensions.append(np.array(v))

            #New scene 200 uncomment here
            all_rotations.append(data["obj_rotations"][k])

            translation = np.array(data['obj_translations'][k] - obj_location)
            all_translations.append(translation)

    return all_c2w_train, focal, img_wh, bbox_dimensions, all_rotations, all_translations

def get_masked_textured_pointclouds(depth,rgb, intrinsics, width, height):
    xmap = np.array([[y for y in range(width)] for z in range(height)])
    ymap = np.array([[z for y in range(width)] for z in range(height)])
    cam_cx = intrinsics[0,2]
    cam_fx = intrinsics[0,0]
    cam_cy = intrinsics[1,2]
    cam_fy = intrinsics[1,1]

    depth_masked = depth.reshape(-1)[:, np.newaxis]
    xmap_masked = xmap.flatten()[:, np.newaxis]
    ymap_masked = ymap.flatten()[:, np.newaxis]
    rgb = rgb.reshape(-1,3)/255.0
    pt2 = depth_masked
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, pt1, pt2), axis=1)
    return points, rgb

def convert_nerf_to_PD(C2W):

    flip_axes = np.array([[1,0,0,0],
                         [0,0,-1,0],
                         [0,1,0,0],
                         [0,0,0,1]])
    C2W = np.matmul(C2W, np.linalg.inv(flip_axes))
    return C2W
                                                       
def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def get_RTs(obj_poses):
    all_boxes = []
    all_translations = []
    all_rotations= []
    for i, bbox in enumerate(obj_poses['bbox_dimensions']):
            all_boxes.append(bbox)
            translation = np.array(obj_poses['obj_translations'][i])
            all_translations.append(translation)
            all_rotations.append(obj_poses["obj_rotations"][i])
    RTs = {'R': all_rotations, 'T': all_translations, 's': all_boxes}
    return RTs

def draw_combined_pcds_boxes(obj_pose, all_depth_pc, all_rgb_pc, all_c2w):
    RTs = get_RTs(obj_pose)
    all_pcd = []
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']
    for Rot,Tran,bbox in zip(all_rotations, all_translations, bbox_dimensions):
        Tran = np.array(Tran)
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)
        sca = bbox
        bbox = o3d.geometry.OrientedBoundingBox(center = Tran, R = Rot, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)

    src_view_num = [7, 50, 66]
    all_depth_pc_src = [all_depth_pc[i] for i in src_view_num]
    all_rgb_pc_src = [all_rgb_pc[i] for i in src_view_num]
    all_c2w_src = [all_c2w[i] for i in src_view_num]

    all_depth_pc_rest = all_depth_pc[100:]
    all_rgb_pc_rest = all_rgb_pc[100:]
    all_c2w_rest = all_c2w[100:]

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    all_pcd.append(sphere)
    all_pcd.append(coordinate_frame)

    for pc, rgb, pose in zip(all_depth_pc_src, all_rgb_pc_src, all_c2w_src):
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pc)
        pcd_vis.colors = o3d.utility.Vector3dVector(rgb)
        pcd_vis.transform(convert_pose(pose))

        # pcd_vis.transform(convert_pose(pose))
        all_pcd.append(pcd_vis)
        # display camera origin
        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.2, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(convert_pose(pose))
        all_pcd.append(FOR_cam)
        frustums = []
        for C2W in all_c2w_src:
            img_size = (640, 480)
            frustums.append(get_camera_frustum(img_size, focal, convert_pose(C2W), frustum_length=1.5, color=[1,0,0]))

        cameras = frustums2lineset(frustums)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(cameras)

    for pc, rgb, pose in zip(all_depth_pc_rest, all_rgb_pc_rest, all_c2w_rest):
        pcd_vis = o3d.geometry.PointCloud()
        
        pcd_vis.points = o3d.utility.Vector3dVector(pc)
        pcd_vis.colors = o3d.utility.Vector3dVector(rgb)
        pcd_vis.transform(convert_pose(pose))
        # pcd_vis.transform(pose)
        all_pcd.append(pcd_vis)
        # display camera origin
        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(convert_pose(pose))
        # FOR_cam.transform(pose)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(FOR_cam)
        frustums = []
        for C2W in all_c2w_rest:
            img_size = (640, 480)
            frustums.append(get_camera_frustum(img_size, focal, convert_pose(C2W), frustum_length=0.5, color=[0,1,0]))
            # frustums.append(get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0,1,0]))

        cameras = frustums2lineset(frustums)
        # PLOT CAMERAS HEREEEE
        all_pcd.append(cameras)
    
    o3d.visualization.draw_geometries(all_pcd)
    # custom_draw_geometry_with_key_callback(all_pcd)

def separate_lists(lst, indices):
    first_list = [lst[i] for i in range(len(lst)) if i in indices]
    second_list = [lst[i] for i in range(len(lst)) if i not in indices]
    return first_list, second_list

def draw_pcd_and_box(depth, rgb_pc, obj_pose, c2w):
    RTs = get_RTs(obj_pose)
    all_pcd = []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(depth.flatten(1).t()))

    # pcd.colors = o3d.utility.Vector3dVector(rgb_pc)
    all_pcd.append(pcd)
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=5)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=np.array([0., 0., 0.]))
    sphere.paint_uniform_color((1, 0, 0))
    all_pcd.append(sphere)
    all_pcd.append(coordinate_frame)

    for Rot,Tran,bbox in zip(all_rotations, all_translations, bbox_dimensions):
        Tran = np.array(Tran)
        box_transform = np.eye(4)
        box_transform[:3,:3] = np.array(Rot)
        box_transform[:3, 3] = np.array(Tran)

        box_transform = np.linalg.inv(convert_pose(c2w)) @ box_transform
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        coordinate_frame.transform(box_transform)

        sca = bbox

        Rot_transformed = box_transform[:3,:3]
        Tran_transformed = box_transform[:3, 3]

        bbox = o3d.geometry.OrientedBoundingBox(center = Tran_transformed, R = Rot_transformed, extent=sca)
        all_pcd.append(bbox)
        all_pcd.append(coordinate_frame)


        frustums = []
        img_size = (640, 480)
        frustums.append(get_camera_frustum(img_size, focal, convert_pose(c2w), frustum_length=1.0, color=[1,0,0]))
        cameras = frustums2lineset(frustums)
        FOR_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
        FOR_cam.transform(convert_pose(c2w))
        all_pcd.append(FOR_cam)
        all_pcd.append(cameras)

    custom_draw_geometry_with_key_callback(all_pcd)

def preprocess_RTS_for_vis(RTS):
    all_R = RTS['R']
    all_T = RTS['T']
    all_s = RTS['s']

    obj_poses = {}
    obj_poses["bbox_dimensions"] = []
    obj_poses["obj_translations"] = []
    obj_poses["obj_rotations"] = []

    for Rot,Tran,sca in zip(all_R, all_T, all_s):
        bbox_extent = np.array([(sca[1,0]-sca[0,0]), (sca[1,1]-sca[0,1]), (sca[1,2]-sca[0,2])])
        cam_t = Tran
        bbox = np.array(sca)
        bbox_diff = bbox[0,2]+((bbox[1,2]-bbox[0,2])/2)
        cam_t[2] += bbox_diff
        cam_rot = np.array(Rot)[:3, :3]

        obj_poses["bbox_dimensions"].append(bbox_extent)
        obj_poses["obj_translations"].append(cam_t)
        obj_poses["obj_rotations"].append(cam_rot)
    return obj_poses

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

    img_files = os.listdir(os.path.join(base_dir, 'train','rgb'))
    img_files.sort()

    all_c2w, focal, img_size, bbox_dimensions, all_rotations, all_translations = read_poses(pose_dir_train = os.path.join(base_dir,'train', 'pose'), img_files_train = img_files)
    all_bboxes = []
    all_T = []
    all_R = []
    RTS_raw = {'R': all_rotations, 'T': all_translations, 's': bbox_dimensions}
    obj_poses = preprocess_RTS_for_vis(RTS_raw)


    depth_folder = os.path.join(base_dir, 'train','depth')
    depth_paths = os.listdir(depth_folder)
    depth_paths.sort()
    all_depths = [np.clip(np.load(os.path.join(depth_folder, depth_path), allow_pickle=True)['arr_0'], 0,100) for depth_path in depth_paths]
    for idx in range(len(all_depths)):
        all_depths[idx][all_depths[idx] >1000.0] = 0.0

    # fov = 80
    # focal = (640 / 2) / np.tan(( fov/ 2) / (180 / np.pi))

    intrinsics = np.array([
            [focal, 0., 640 / 2.0],
            [0., focal, 480 / 2.0],
            [0., 0., 1.],
        ])
    image_folder = os.path.join(base_dir, 'train','rgb')
    image_paths = os.listdir(image_folder)
    image_paths.sort()

    all_rgbs = [np.array(Image.open(os.path.join(image_folder, img_path))) for img_path in image_paths]

    c2w = all_c2w[0]
    
    K_matrix = np.eye(4)
    K_matrix[:3,:3] = intrinsics
    im = all_rgbs[0]
    box_obb = []
    axes = []

    RTs = get_RTs(obj_poses)
    
    all_rotations = RTs['R']
    all_translations = RTs['T']
    bbox_dimensions = RTs['s']

    all_depth_pc = []
    all_rgb_pc = []
    for depth, rgb in zip(all_depths, all_rgbs):
        depth_pc, rgb_pc = get_masked_textured_pointclouds(depth, rgb, intrinsics, width=640, height=480)
        all_depth_pc.append(depth_pc)
        all_rgb_pc.append(rgb_pc)

    pcd = o3d.geometry.PointCloud()
    H = 480
    W = 640

    rgb_pc = all_rgb_pc[0]
    depth_pc = all_depth_pc[0]

    xyz_orig = torch.from_numpy(depth_pc).reshape(H,W,3).permute(2,0,1)

    draw_combined_pcds_boxes(obj_poses, all_depth_pc, all_rgb_pc, all_c2w)