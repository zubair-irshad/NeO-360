import cv2
import numpy as np
import torch
from torchvision import transforms
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import functools
import math
import warnings

def get_world_grid(side_lengths, grid_size, device):
    """ Returns a 3D grid of points in world coordinates.
    :param side_lengths: (min, max) for each axis (3, 2)
    :param grid_size: number of points along each dimension () or (3)
    :output grid: (1, grid_size**3, 3)
    """
    if len(grid_size) == 1:
        grid_size = [grid_size[0] for _ in range(3)]
        
    w_x = torch.linspace(side_lengths[0][0], side_lengths[0][1], grid_size[0], device = device)
    w_y = torch.linspace(side_lengths[1][0], side_lengths[1][1], grid_size[1], device = device)
    w_z = torch.linspace(side_lengths[2][0], side_lengths[2][1], grid_size[2], device = device)
    X, Y, Z = torch.meshgrid(w_x, w_y, w_z)
    w_xyz = torch.stack([X, Y, Z], axis=-1).reshape(-1, 3).unsqueeze(0) # (gs, gs, gs, 3), gs = grid_size
    return w_xyz

def world2camera_rot(w_xyz, cam2world, NS):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = repeat_interleave(w_xyz, NS)  # (SB*NS, B, 3)
    rot = cam2world[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    # trans = -torch.bmm(rot, cam2world[:, :3, 3:])  # (B, 3, 1)
    cam_rot = torch.matmul(rot[:, None, :3, :3], w_xyz.unsqueeze(-1))[..., 0]
    # cam_xyz = cam_rot + trans[:, None, :, 0]
    return cam_rot


def world2camera_viewdirs(w_viewdirs, cam2world, NS):
    w_viewdirs = repeat_interleave(w_viewdirs, NS)  # (SB*NS, B, 3)
    rot = cam2world[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    viewdirs = torch.matmul(rot[:, None, :3, :3], w_viewdirs.unsqueeze(-1))[..., 0]
    return viewdirs


def world2camera(w_xyz, cam2world, NS=None):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    #print(w_xyz.shape, cam2world.shape)
    if NS is not None:
        w_xyz = repeat_interleave(w_xyz, NS)  # (SB*NS, B, 3)
    rot = cam2world[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
    trans = -torch.bmm(rot, cam2world[:, :3, 3:])  # (B, 3, 1)
    #print(rot.shape, w_xyz.shape)
    cam_rot = torch.matmul(rot[:, None, :3, :3], w_xyz.unsqueeze(-1))[..., 0]
    cam_xyz = cam_rot + trans[:, None, :, 0]
    # cam_xyz = cam_xyz.reshape(-1, 3)  # (SB*B, 3)
    return cam_xyz


def w2i_projection(w_xyz, cam2world, intrinsics):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = torch.cat([w_xyz, torch.ones_like(w_xyz[..., :1])], dim=-1)  # [n_points, 4]
    cam_xyz = torch.inverse(cam2world).bmm(w_xyz.permute(0,2,1))
    camera_grids = cam_xyz.permute(0,2,1)[:,:,:3]
    mask = camera_grids[..., 2] < 1e-3
    projections = intrinsics[None, ...].repeat(cam2world.shape[0], 1, 1).bmm(cam_xyz[:,:3,:])
    projections = projections.permute(0,2,1)
    uv = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
    uv = torch.clamp(uv, min=-1e6, max=1e6)
    return camera_grids, uv, mask

def projection(c_xyz, focal, c, NV=None):
    """Converts [x,y,z] in camera coordinates to image coordinates 
        for the given focal length focal and image center c.
    :param c_xyz: points in camera coordinates (SB*NV, NP, 3)
    :param focal: focal length (SB, 2)
    :c: image center (SB, 2)
    :output uv: pixel coordinates (SB, NV, NP, 2)
    """
    
    if NV is None:
        NV = int(c_xyz.shape[0]/c.shape[0])

    uv = -c_xyz[..., :2] / (c_xyz[..., 2:] + 1e-9)  # (SB*NV, NC, 2); NC: number of grid cells 
    uv *= repeat_interleave(
                focal.unsqueeze(1), NV if focal.shape[0] > 1 else 1
            )
    uv += repeat_interleave(
                c.unsqueeze(1), NV if c.shape[0] > 1 else 1
            )
    return uv

def combine_interleaved(t, inner_dims=(1,), agg_type="agg_type"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t

def image_float_to_uint8(img):
    """
    Convert a float image (0.0-1.0) to uint8 (0-255)
    """
    vmin = np.min(img)
    vmax = np.max(img)
    if vmax - vmin < 1e-10:
        vmax += 1e-10
    img = (img - vmin) / (vmax - vmin)
    img *= 255.0
    return img.astype(np.uint8)

def convert_pose(C2W):
    flip_yz = torch.eye(4).to(C2W.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = torch.matmul(C2W, flip_yz)
    return C2W

def w2i_projection(w_xyz, cam2world, intrinsics):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = torch.cat([w_xyz, torch.ones_like(w_xyz[..., :1])], dim=-1)  # [n_points, 4]
    cam_xyz = torch.inverse(cam2world).bmm(w_xyz.permute(0,2,1))
    camera_grids = cam_xyz.permute(0,2,1)[:,:,:3]
    projections = intrinsics[None, ...].repeat(cam2world.shape[0], 1, 1).bmm(cam_xyz[:,:3,:])
    projections = projections.permute(0,2,1)
    im_z = projections[..., 2:3]
    uv = projections[..., :2] / projections[..., 2:3]  # [n_views, n_points, 2]
    return camera_grids, uv, im_z



def cmap(img, color_map=cv2.COLORMAP_HOT):
    """
    Apply 'HOT' color to a float image
    """
    return cv2.applyColorMap(image_float_to_uint8(img), color_map)


def batched_index_select_nd(t, inds):
    """
    Index select on dim 1 of a n-dimensional batched tensor.
    :param t (batch, n, ...)
    :param inds (batch, k)
    :return (batch, k, ...)
    """
    return t.gather(
        1, inds[(...,) + (None,) * (len(t.shape) - 2)].expand(-1, -1, *t.shape[2:])
    )


def batched_index_select_nd_last(t, inds):
    """
    Index select on dim -1 of a >=2D multi-batched tensor. inds assumed
    to have all batch dimensions except one data dimension 'n'
    :param t (batch..., n, m)
    :param inds (batch..., k)
    :return (batch..., n, k)
    """
    dummy = inds.unsqueeze(-2).expand(*inds.shape[:-1], t.size(-2), inds.size(-1))
    out = t.gather(-1, dummy)
    return out


def repeat_interleave(input, repeats, dim=0):
    """
    Repeat interleave along axis 0
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    output = input.unsqueeze(1).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
    return transforms.Compose(ops)


def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )


def homogeneous(points):
    """
    Concat 1 to each point
    :param points (..., 3)
    :return (..., 4)
    """
    return F.pad(points, (0, 1), "constant", 1.0)


def gen_grid(*args, ij_indexing=False):
    """
    Generete len(args)-dimensional grid.
    Each arg should be (lo, hi, sz) so that in that dimension points
    are taken at linspace(lo, hi, sz).
    Example: gen_grid((0,1,10), (-1,1,20))
    :return (prod_i args_i[2], len(args)), len(args)-dimensional grid points
    """
    return torch.from_numpy(
        np.vstack(
            np.meshgrid(
                *(np.linspace(lo, hi, sz, dtype=np.float32) for lo, hi, sz in args),
                indexing="ij" if ij_indexing else "xy"
            )
        )
        .reshape(len(args), -1)
        .T
    )


def unproj_map(width, height, f, c=None, device="cpu"):
    """
    Get camera unprojection map for given image size.
    [y,x] of output tensor will contain unit vector of camera ray of that pixel.
    :param width image width
    :param height image height
    :param f focal length, either a number or tensor [fx, fy]
    :param c principal point, optional, either None or tensor [fx, fy]
    if not specified uses center of image
    :return unproj map (height, width, 3)
    """
    if c is None:
        c = [width * 0.5, height * 0.5]
    else:
        c = c.squeeze()
    if isinstance(f, float):
        f = [f, f]
    elif len(f.shape) == 0:
        f = f[None].expand(2)
    elif len(f.shape) == 1:
        f = f.expand(2)
    Y, X = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) - float(c[1]),
        torch.arange(width, dtype=torch.float32) - float(c[0]),
    )
    X = X.to(device=device) / float(f[0])
    Y = Y.to(device=device) / float(f[1])
    Z = torch.ones_like(X)
    unproj = torch.stack((X, -Y, -Z), dim=-1)
    unproj /= torch.norm(unproj, dim=-1).unsqueeze(-1)
    return unproj


def coord_from_blender(dtype=torch.float32, device="cpu"):
    """
    Blender to standard coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def coord_to_blender(dtype=torch.float32, device="cpu"):
    """
    Standard to Blender coordinate system transform.
    Standard coordinate system is: x right y up z out (out=screen to face)
    Blender coordinate system is: x right y in z up
    :return (4, 4)
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=dtype,
        device=device,
    )


def look_at(origin, target, world_up=np.array([0, 1, 0], dtype=np.float32)):
    """
    Get 4x4 camera to world space matrix, for camera looking at target
    """
    back = origin - target
    back /= np.linalg.norm(back)
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)

    cam_to_world = np.empty((4, 4), dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = back
    cam_to_world[:3, 3] = origin
    cam_to_world[3, :] = [0, 0, 0, 1]
    return cam_to_world


def get_cuda(gpu_id):
    """
    Get a torch.device for GPU gpu_id. If GPU not available,
    returns CPU device.
    """
    return (
        torch.device("cuda:%d" % gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def masked_sample(masks, num_pix, prop_inside, thresh=0.5):
    """
    :return (num_pix, 3)
    """
    num_inside = int(num_pix * prop_inside + 0.5)
    num_outside = num_pix - num_inside
    inside = (masks >= thresh).nonzero(as_tuple=False)
    outside = (masks < thresh).nonzero(as_tuple=False)

    pix_inside = inside[torch.randint(0, inside.shape[0], (num_inside,))]
    pix_outside = outside[torch.randint(0, outside.shape[0], (num_outside,))]
    pix = torch.cat((pix_inside, pix_outside))
    return pix


def bbox_sample(bboxes, num_pix):
    """
    :return (num_pix, 3)
    """
    image_ids = torch.randint(0, bboxes.shape[0], (num_pix,))
    pix_bboxes = bboxes[image_ids]
    x = (
        torch.rand(num_pix) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0])
        + pix_bboxes[:, 0]
    ).long()
    y = (
        torch.rand(num_pix) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1])
        + pix_bboxes[:, 1]
    ).long()
    pix = torch.stack((image_ids, y, x), dim=-1)
    return pix


def gen_rays(poses, width, height, focal, z_near, z_far, c=None, ndc=False):
    """
    Generate camera rays
    :return (B, H, W, 8)
    """
    num_images = poses.shape[0]
    device = poses.device
    cam_unproj_map = (
        unproj_map(width, height, focal.squeeze(), c=c, device=device)
        .unsqueeze(0)
        .repeat(num_images, 1, 1, 1)
    )
    cam_centers = poses[:, None, None, :3, 3].expand(-1, height, width, -1)
    cam_raydir = torch.matmul(
        poses[:, None, None, :3, :3], cam_unproj_map.unsqueeze(-1)
    )[:, :, :, :, 0]
    if ndc:
        if not (z_near == 0 and z_far == 1):
            warnings.warn(
                "dataset z near and z_far not compatible with NDC, setting them to 0, 1 NOW"
            )
        z_near, z_far = 0.0, 1.0
        cam_centers, cam_raydir = ndc_rays(
            width, height, focal, 1.0, cam_centers, cam_raydir
        )

    cam_nears = (
        torch.tensor(z_near, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    cam_fars = (
        torch.tensor(z_far, device=device)
        .view(1, 1, 1, 1)
        .expand(num_images, height, width, -1)
    )
    return torch.cat(
        (cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1
    )  # (B, H, W, 8)


def trans_t(t):
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=torch.float32,
    )


def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def rot_theta(th):
    return torch.tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.tensor(
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        @ c2w
    )
    return c2w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def make_conv_2d(
    dim_in,
    dim_out,
    padding_type="reflect",
    norm_layer=None,
    activation=None,
    kernel_size=3,
    use_bias=False,
    stride=1,
    no_pad=False,
    zero_init=False,
):
    conv_block = []
    amt = kernel_size // 2
    if stride > 1 and not no_pad:
        raise NotImplementedError(
            "Padding with stride > 1 not supported, use same_pad_conv2d"
        )

    if amt > 0 and not no_pad:
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(amt)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(amt)]
        elif padding_type == "zero":
            conv_block += [nn.ZeroPad2d(amt)]
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

    conv_block.append(
        nn.Conv2d(
            dim_in, dim_out, kernel_size=kernel_size, bias=use_bias, stride=stride
        )
    )
    if zero_init:
        nn.init.zeros_(conv_block[-1].weight)
    #  else:
    #  nn.init.kaiming_normal_(conv_block[-1].weight)
    if norm_layer is not None:
        conv_block.append(norm_layer(dim_out))

    if activation is not None:
        conv_block.append(activation)
    return nn.Sequential(*conv_block)


def calc_same_pad_conv2d(t_shape, kernel_size=3, stride=1):
    in_height, in_width = t_shape[-2:]
    out_height = math.ceil(in_height / stride)
    out_width = math.ceil(in_width / stride)

    pad_along_height = max((out_height - 1) * stride + kernel_size - in_height, 0)
    pad_along_width = max((out_width - 1) * stride + kernel_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def same_pad_conv2d(t, padding_type="reflect", kernel_size=3, stride=1, layer=None):
    """
    Perform SAME padding on tensor, given kernel size/stride of conv operator
    assumes kernel/stride are equal in all dimensions.
    Use before conv called.
    Dilation not supported.
    :param t image tensor input (B, C, H, W)
    :param padding_type padding type constant | reflect | replicate | circular
    constant is 0-pad.
    :param kernel_size kernel size of conv
    :param stride stride of conv
    :param layer optionally, pass conv layer to automatically get kernel_size and stride
    (overrides these)
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    return F.pad(
        t, calc_same_pad_conv2d(t.shape, kernel_size, stride), mode=padding_type
    )


def same_unpad_deconv2d(t, kernel_size=3, stride=1, layer=None):
    """
    Perform SAME unpad on tensor, given kernel/stride of deconv operator.
    Use after deconv called.
    Dilation not supported.
    """
    if layer is not None:
        if isinstance(layer, nn.Sequential):
            layer = next(layer.children())
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
    h_scaled = (t.shape[-2] - 1) * stride
    w_scaled = (t.shape[-1] - 1) * stride
    pad_left, pad_right, pad_top, pad_bottom = calc_same_pad_conv2d(
        (h_scaled, w_scaled), kernel_size, stride
    )
    if pad_right == 0:
        pad_right = -10000
    if pad_bottom == 0:
        pad_bottom = -10000
    return t[..., pad_top:-pad_bottom, pad_left:-pad_right]


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


def psnr(pred, target):
    """
    Compute PSNR of two tensors in decibels.
    pred/target should be of same size or broadcastable
    """
    mse = ((pred - target) ** 2).mean()
    psnr = -10 * math.log10(mse)
    return psnr


def quat_to_rot(q):
    """
    Quaternion to rotation matrix
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3, 3), device=q.device)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (qj ** 2 + qk ** 2)
    R[:, 0, 1] = 2 * (qj * qi - qk * qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1 - 2 * (qi ** 2 + qk ** 2)
    R[:, 1, 2] = 2 * (qj * qk - qi * qr)
    R[:, 2, 0] = 2 * (qk * qi - qj * qr)
    R[:, 2, 1] = 2 * (qj * qk + qi * qr)
    R[:, 2, 2] = 1 - 2 * (qi ** 2 + qj ** 2)
    return R


def rot_to_quat(R):
    """
    Rotation matrix to quaternion
    """
    batch_size, _, _ = R.shape
    q = torch.ones((batch_size, 4), device=R.device)

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:, 0] = torch.sqrt(1.0 + R00 + R11 + R22) / 2
    q[:, 1] = (R21 - R12) / (4 * q[:, 0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_module(net):
    """
    Shorthand for either net.module (if net is instance of DataParallel) or net
    """
    if isinstance(net, torch.nn.DataParallel):
        return net.module
    else:
        return net
    
from kornia.utils import create_meshgrid

def homo_warp(src_feat, proj_mat, depth_values, src_grid=None, pad=0):
    """
    src_feat: (B, C, H, W)
    proj_mat: (B, 3, 4) equal to "src_proj @ ref_proj_inv"
    depth_values: (B, D, H, W)
    out: (B, C, D, H, W)
    """

    if src_grid==None:
        B, C, H, W = src_feat.shape
        device = src_feat.device

        if pad>0:
            H_pad, W_pad = H + pad*2, W + pad*2
        else:
            H_pad, W_pad = H, W

        depth_values = depth_values[...,None,None].repeat(1, 1, H_pad, W_pad)
        D = depth_values.shape[1]

        R = proj_mat[:, :, :3]  # (B, 3, 3)
        T = proj_mat[:, :, 3:]  # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_pad, W_pad, normalized_coordinates=False, device=device)  # (1, H, W, 2)
        if pad>0:
            ref_grid -= pad

        ref_grid = ref_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, W_pad * H_pad)  # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1)  # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:, :1])), 1)  # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D)  # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T / depth_values.view(B, 1, D * W_pad * H_pad)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values  # release (GPU) memory



        src_grid = src_grid_d[:, :2] / src_grid_d[:, 2:]  # divide by depth (B, 2, D*H*W)
        del src_grid_d
        src_grid[:, 0] = src_grid[:, 0] / ((W - 1) / 2) - 1  # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H - 1) / 2) - 1  # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, W_pad, H_pad, 2)

    B, D, W_pad, H_pad = src_grid.shape[:4]
    warped_src_feat = F.grid_sample(src_feat, src_grid.view(B, D, W_pad * H_pad, 2),
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True)  # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, -1, D, H_pad, W_pad)
    # src_grid = src_grid.view(B, 1, D, H_pad, W_pad, 2)
    return warped_src_feat, src_grid

def projection_extrinsics(w_xyz, w2c, intrinsics):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = torch.cat([w_xyz, torch.ones_like(w_xyz[..., :1])], dim=-1)  # [n_points, 4]
    cam_xyz = w2c.bmm(w_xyz.permute(0,2,1))
    camera_grids = cam_xyz.permute(0,2,1)[:,:,:3]
    projections = intrinsics[None, ...].repeat(w2c.shape[0], 1, 1).bmm(cam_xyz[:,:3,:])
    projections = projections.permute(0,2,1)
    
    uv = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
    #uv = projections[..., :2] / projections[..., 2:3]  # [n_views, n_points, 2]
    mask = projections[..., 2] > 0
    return camera_grids, uv, mask

def projection_extrinsics_alldim(w_xyz, w2c, intrinsics):
    """Converts the points in world coordinates to camera view.
    :param xyz: points in world coordinates (SB*NV, NC, 3)
    :param poses: camera matrix (SB*NV, 4, 4)
    :output points in camera coordinates (SB*NV, NC, 3)
    : SB batch size
    : NV number of views in each scene
    : NC number of coordinate points
    """
    w_xyz = torch.cat([w_xyz, torch.ones_like(w_xyz[..., :1])], dim=-1)  # [n_points, 4]
    cam_xyz = w2c.bmm(w_xyz.permute(0,2,1))
    camera_grids = cam_xyz.permute(0,2,1)[:,:,:3]
    projections = intrinsics[None, ...].repeat(w2c.shape[0], 1, 1).bmm(cam_xyz[:,:3,:])
    projections = projections.permute(0,2,1)
    return projections

def get_ndc_samples(samples, w2c, focal, c, H, W, near=0.2, far = 2.5):
    """
    Get pixel-aligned image features at 2D image coordinates
    :param uv (B, N, 2) image points (x,y)
    :param cam_z ignored (for compatibility)
    :param image_size image size, either (width, height) or single int.
    if not specified, assumes coords are in [-1, 1]
    :param z_bounds ignored (for compatibility)
    :return (B, L, N) L is latent size
    """ 

    #0th is the reference view
    B, N_samples, _ = samples.shape
    samples = samples.reshape(-1,3).unsqueeze(0)

    w2c_ref = w2c[0].unsqueeze(0)
    inv_scale = torch.tensor([W-1, H-1]).to(samples.device)

    intrinsics_ref = torch.FloatTensor([
        [focal[0], 0., c[0][0]],
        [0., focal[0], c[0][1]],
        [0., 0., 1.],
        ]).to(w2c_ref.device)
    #our feature size is H/2, W/2
    intrinsics_ref = intrinsics_ref/2
    intrinsics_ref[-1,-1] = 1

    if intrinsics_ref is not None:
        point_samples_pixel = projection_extrinsics_alldim(samples, w2c_ref, intrinsics_ref)
        point_samples_pixel = point_samples_pixel.squeeze(0)
        point_samples_pixel[:,:2] = (point_samples_pixel[:,:2] / point_samples_pixel[:,-1:] + 0.0) / inv_scale.reshape(1,2)  
        point_samples_pixel[:,2] = (point_samples_pixel[:,2] - near) / (far - near)  # normalize to 0~1
    point_samples_pixel = point_samples_pixel.reshape(B, N_samples, 3)
    return point_samples_pixel

def gen_dir_feature(w2c, rays_dir):
    """
    Inputs:
        c2ws: [1,v,4,4]
        rays_pts: [N_rays, N_samples, 3]
        rays_dir: [N_rays, 3]
    Returns:
    """
    w2c_ref = w2c[0]
    dirs = rays_dir @ w2c_ref[:3,:3].t() # [N_rays, 3]
    return dirs


