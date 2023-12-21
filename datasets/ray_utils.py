import torch
from kornia import create_meshgrid
import numpy as np
import matplotlib.pyplot as plt
import numba as nb


def homogenise_np(p):
    _1 = np.ones((p.shape[0], 1), dtype=p.dtype)
    return np.concatenate([p, _1], axis=-1)


def inside_axis_aligned_box(pts, box_min, box_max):
    return torch.all(torch.cat([pts >= box_min, pts <= box_max], dim=1), dim=1)


@nb.jit(nopython=True)
def bbox_intersection_batch(bounds, rays_o, rays_d):
    N_rays = rays_o.shape[0]
    all_hit = np.empty((N_rays))
    all_near = np.empty((N_rays))
    all_far = np.empty((N_rays))
    for idx, (o, d) in enumerate(zip(rays_o, rays_d)):
        hit, near, far = bbox_intersection(bounds, o, d)
        # if hit == True:
        #     print("hit", hit)
        all_hit[idx] = hit
        all_near[idx] = near
        all_far[idx] = far
    # return (h*w), (h*w, 3), (h*w, 3)
    return all_hit, all_near, all_far


@nb.jit(nopython=True)
def bbox_intersection(bounds, orig, dir):
    # FIXME: currently, it is not working properly if the ray origin is inside the bounding box
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    # handle divide by zero
    dir[dir == 0] = 1.0e-14
    invdir = 1 / dir
    sign = (invdir < 0).astype(np.int64)

    tmin = (bounds[sign[0]][0] - orig[0]) * invdir[0]
    tmax = (bounds[1 - sign[0]][0] - orig[0]) * invdir[0]

    tymin = (bounds[sign[1]][1] - orig[1]) * invdir[1]
    tymax = (bounds[1 - sign[1]][1] - orig[1]) * invdir[1]

    if tmin > tymax or tymin > tmax:
        return False, 0, 0
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (bounds[sign[2]][2] - orig[2]) * invdir[2]
    tzmax = (bounds[1 - sign[2]][2] - orig[2]) * invdir[2]

    if tmin > tzmax or tzmin > tmax:
        return False, 0, 0
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax
    # additionally, when the orig is inside the box, we return False
    if tmin < 0 or tmax < 0:
        return False, 0, 0
    return True, tmin, tmax


def homogenise_torch(p):
    _1 = torch.ones_like(p[:, [0]])
    return torch.cat([p, _1], dim=-1)


def openCV_to_OpenGL(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


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
    directions = torch.stack(
        [(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1
    )  # (H, W, 3)

    return directions


def get_rays_background(directions, c2w, coords):
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
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_o = rays_o[coords[:, 0], coords[:, 1]]
    rays_d = rays_d[coords[:, 0], coords[:, 1]]

    return rays_o, rays_d


def get_rays(directions, c2w, output_view_dirs=False, output_radii=False):
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
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    # rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    if output_radii:
        rays_d_orig = directions @ c2w[:, :3].T
        dx = torch.sqrt(
            torch.sum((rays_d_orig[:-1, :, :] - rays_d_orig[1:, :, :]) ** 2, dim=-1)
        )
        dx = torch.cat([dx, dx[-2:-1, :]], dim=0)
        radius = dx[..., None] * 2 / torch.sqrt(torch.tensor(12, dtype=torch.int8))
        radius = radius.reshape(-1)

    if output_view_dirs:
        viewdirs = rays_d
        viewdirs /= torch.norm(viewdirs, dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        viewdirs = viewdirs.view(-1, 3)
        if output_radii:
            return rays_o, viewdirs, rays_d, radius
        else:
            return rays_o, viewdirs, rays_d
    else:
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d


def transform_rays_camera(rays_o, rays_d, c2w):
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
    rays_d = rays_d @ c2w[:, :3].T  # (H, W, 3)
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) + rays_o  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * ox_oz
    o1 = -1.0 / (H / (2.0 * focal)) * oy_oz
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def world_to_ndc(rotated_pcds, W, H, focal, near):
    ox_oz = rotated_pcds[..., 0] / rotated_pcds[..., 2]
    oy_oz = rotated_pcds[..., 1] / rotated_pcds[..., 2]

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * ox_oz
    o1 = -1.0 / (H / (2.0 * focal)) * oy_oz
    o2 = 1.0 + 2.0 * near / rotated_pcds[..., 2]

    # oz_ox = rotated_pcds[...,2] / rotated_pcds[...,0]
    # oz_oy = rotated_pcds[...,2] / rotated_pcds[...,1]
    # # Projection
    # o0 = -(W/(2.*focal)) * oz_ox
    # o1 = -(H/(2.*focal)) * oz_oy
    # o2 = 1. + 2. * near / rotated_pcds[...,2]
    # print("o1.shape", o1.shape)
    rotated_pcd = np.concatenate(
        (
            np.expand_dims(o0, axis=-1),
            np.expand_dims(o1, axis=-1),
            np.expand_dims(o2, axis=-1),
        ),
        -1,
    )
    return rotated_pcd


def get_rays_segmented(masks, class_ids, rays_o, rays_d, W, H, N_rays):
    seg_mask = np.zeros([H, W])
    for i in range(len(class_ids)):
        seg_mask[masks[:, :, i] > 0] = np.array(class_ids)[i]
    # print("classIds", class_ids)
    # print("seg masks", (seg_mask>0).flatten().shape, (seg_mask>0).shape)
    # print("(seg_mask>0).flatten()", np.count_nonzero((seg_mask>0).flatten()))
    # print("seg mask ", np.count_nonzero(seg_mask))
    # print("(seg_mask>0).flatten()", (seg_mask>0).flatten())
    # print("seg mask", seg_mask)
    # plt.imshow(seg_mask)
    # plt.show()

    rays_rgb_obj = []
    rays_rgb_obj_dir = []
    class_ids.sort()

    select_inds = []
    for i in range(len(class_ids)):
        rays_on_obj = np.where(seg_mask.flatten() == class_ids[i])[0]
        print("rays_on_obj", rays_on_obj.shape)
        rays_on_obj = rays_on_obj[np.random.choice(rays_on_obj.shape[0], N_rays)]
        select_inds.append(rays_on_obj)
        obj_mask = np.zeros(len(rays_o), np.bool)
        obj_mask[rays_on_obj] = 1
        rays_rgb_obj.append(rays_o[obj_mask])
        rays_rgb_obj_dir.append(rays_d[obj_mask])
    select_inds = np.concatenate(select_inds, axis=0)
    obj_mask = np.zeros(len(rays_o), np.bool)
    obj_mask[select_inds] = 1

    # for i in range(len(class_ids)):
    #     rays_on_obj = np.where(seg_mask.flatten() == class_ids[i])[0]
    #     obj_mask = np.zeros(len(rays_o), np.bool)
    #     obj_mask[rays_on_obj] = 1
    #     rays_rgb_obj.append(rays_o[obj_mask])
    #     rays_rgb_obj_dir.append(rays_d[obj_mask])

    # N_rays = min(N_rays, H * W)
    # select_inds = []
    # for b in range(len(class_ids)):
    #     fg_inds = np.nonzero(seg_mask.flatten() == class_ids[b])
    #     fg_inds = np.transpose(np.asarray(fg_inds))
    #     fg_inds = fg_inds[np.random.choice(fg_inds.shape[0], N_rays)]
    #     select_inds.append(fg_inds)
    # select_inds = np.concatenate(select_inds, axis=0)
    # j, i = select_inds[..., 0], select_inds[..., 1]

    # select_inds = j * W + i

    return rays_rgb_obj, rays_rgb_obj_dir, class_ids, (seg_mask > 0).flatten()


def convert_pose_PD_to_NeRF(C2W):
    flip_axes = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    C2W = np.matmul(C2W, flip_axes)
    return C2W


def get_rays_mvs(H, W, focal, c2w):
    ys, xs = torch.meshgrid(
        torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)
    )  # pytorch's meshgrid has indexing='ij'
    ys, xs = ys.reshape(-1), xs.reshape(-1)

    dirs = torch.stack(
        [(xs - W / 2) / focal, (ys - H / 2) / focal, torch.ones_like(xs)], -1
    )  # use 1 instead of -1
    rays_d = (
        dirs @ c2w[:3, :3].t()
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)
    return rays_o, rays_d
