# ------------------------------------------------------------------------------------
# NeO-360
# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# ------------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals[..., None] * directions[..., None, :]


# import vren
from torch.cuda.amp import custom_fwd, custom_bwd


@torch.jit.script
def grid_sample2d(image, grid):
    """Implements grid_sample2d with double-differentiation support.
    Equivalent to F.grid_sample(..., mode='bilinear',
                                padding_mode='border', align_corners=True).
    """
    bs, nc, ih, iw = image.shape
    _, h, w, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]

    ix = ((ix + 1) / 2) * (iw - 1)
    iy = ((iy + 1) / 2) * (ih - 1)

    ix_nw = torch.floor(ix)
    iy_nw = torch.floor(iy)
    ix_ne = ix_nw + 1
    iy_ne = iy_nw
    ix_sw = ix_nw
    iy_sw = iy_nw + 1
    ix_se = ix_nw + 1
    iy_se = iy_nw + 1

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    ix_nw = torch.clamp(ix_nw.long(), 0, iw - 1)
    iy_nw = torch.clamp(iy_nw.long(), 0, ih - 1)

    ix_ne = torch.clamp(ix_ne.long(), 0, iw - 1)
    iy_ne = torch.clamp(iy_ne.long(), 0, ih - 1)

    ix_sw = torch.clamp(ix_sw.long(), 0, iw - 1)
    iy_sw = torch.clamp(iy_sw.long(), 0, ih - 1)

    ix_se = torch.clamp(ix_se.long(), 0, iw - 1)
    iy_se = torch.clamp(iy_se.long(), 0, ih - 1)

    image = image.view(bs, nc, ih * iw)

    nw_val = torch.gather(
        image, 2, (iy_nw * iw + ix_nw).view(bs, 1, h * w).expand(-1, nc, -1)
    )
    ne_val = torch.gather(
        image, 2, (iy_ne * iw + ix_ne).view(bs, 1, h * w).expand(-1, nc, -1)
    )
    sw_val = torch.gather(
        image, 2, (iy_sw * iw + ix_sw).view(bs, 1, h * w).expand(-1, nc, -1)
    )
    se_val = torch.gather(
        image, 2, (iy_se * iw + ix_se).view(bs, 1, h * w).expand(-1, nc, -1)
    )

    out_val = (
        nw_val.view(bs, nc, h, w) * nw.view(bs, 1, h, w)
        + ne_val.view(bs, nc, h, w) * ne.view(bs, 1, h, w)
        + sw_val.view(bs, nc, h, w) * sw.view(bs, 1, h, w)
        + se_val.view(bs, nc, h, w) * se.view(bs, 1, h, w)
    )

    return out_val


def get_ray_limits(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length=2):
    batch_near, batch_far = get_ray_limits_box(
        rays_o, rays_d, box_side_length=box_side_length
    )
    is_ray_valid = batch_far > batch_near
    if torch.any(is_ray_valid).item():
        batch_near[~is_ray_valid] = batch_near[is_ray_valid].min()
        batch_far[~is_ray_valid] = batch_far[is_ray_valid].max()
    batch_near[batch_near < 0] = 0
    batch_far[batch_far < 0] = 0
    return batch_near, batch_far


def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)
    bb_min = [
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
    ]
    bb_max = [
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
    ]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)
    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()
    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[
        ..., 1
    ]
    tymax = (
        bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]
    ) * invdir[..., 1]
    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False
    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)
    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[
        ..., 2
    ]
    tzmax = (
        bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]
    ) * invdir[..., 2]
    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False
    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)
    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2
    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)


class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.
    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)
    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


def get_z_vals(
    rays_o,
    rays_d,
    near,
    far,
    randomized,
    model,
    lindisp,
    N_samples=64,
    N_samples_eval=128,
    N_samples_extra=32,
    eps=0.1,
    beta_iters=10,
    max_total_iters=5,
    add_tiny=0.0,
):
    beta0 = model.density.get_beta().detach()

    # Start with uniform sampling
    z_vals, _ = sample_along_rays(
        rays_o,
        rays_d,
        num_samples=N_samples_eval,
        near=near,
        far=far,
        randomized=randomized,
        lindisp=lindisp,
    )

    samples, samples_idx = z_vals, None

    # Get maximum beta from the upper bound (Lemma 2)
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    bound = (1.0 / (4.0 * torch.log(torch.tensor(eps + 1.0)))) * (dists**2.0).sum(-1)
    beta = torch.sqrt(bound)

    total_iters, not_converge = 0, True

    # Algorithm 1
    while not_converge and total_iters < max_total_iters:
        points = rays_o.unsqueeze(1) + samples.unsqueeze(2) * rays_d.unsqueeze(1)
        points_flat = points

        # print("points", points.shape)
        # print("samples", samples.shape)
        # Calculating the SDF only for the new sampled points
        with torch.no_grad():
            samples_sdf = model.get_sdf_vals(points_flat)
        if samples_idx is not None:
            sdf_merge = torch.cat(
                [
                    sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                    samples_sdf.reshape(-1, samples.shape[1]),
                ],
                -1,
            )
            sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
        else:
            sdf = samples_sdf

        # Calculating the bound d* (Theorem 1)
        d = sdf.reshape(z_vals.shape)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).to(rays_o.device)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

        # Updating beta using line search
        curr_error = get_error_bound(beta0, model, sdf, z_vals, dists, d_star)
        beta[curr_error <= eps] = beta0
        beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
        for j in range(beta_iters):
            beta_mid = (beta_min + beta_max) / 2.0
            curr_error = get_error_bound(
                beta_mid.unsqueeze(-1), model, sdf, z_vals, dists, d_star
            )
            beta_max[curr_error <= eps] = beta_mid[curr_error <= eps]
            beta_min[curr_error > eps] = beta_mid[curr_error > eps]
        beta = beta_max

        # Upsample more points
        density = model.density(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

        dists = torch.cat(
            [
                dists,
                torch.tensor([1e10])
                .to(rays_o.device)
                .unsqueeze(0)
                .repeat(dists.shape[0], 1),
            ],
            -1,
        )
        free_energy = dists * density
        shifted_free_energy = torch.cat(
            [torch.zeros(dists.shape[0], 1).to(rays_o.device), free_energy[:, :-1]],
            dim=-1,
        )
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here

        #  Check if we are done and this is the last sampling
        total_iters += 1
        not_converge = beta.max() > beta0

        if not_converge and total_iters < max_total_iters:
            """Sample more points proportional to the current error bound"""

            N = N_samples_eval

            bins = z_vals
            error_per_section = (
                torch.exp(-d_star / beta.unsqueeze(-1))
                * (dists[:, :-1] ** 2.0)
                / (4 * beta.unsqueeze(-1) ** 2)
            )
            error_integral = torch.cumsum(error_per_section, dim=-1)
            bound_opacity = (
                torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
            ) * transmittance[:, :-1]

            pdf = bound_opacity + add_tiny
            pdf = pdf / torch.sum(pdf, -1, keepdim=True)
            cdf = torch.cumsum(pdf, -1)
            cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        else:
            """Sample the final sample set to be used in the volume rendering integral"""

            N = N_samples

            bins = z_vals
            pdf = weights[..., :-1]
            pdf = pdf + 1e-5  # prevent nans
            pdf = pdf / torch.sum(pdf, -1, keepdim=True)
            cdf = torch.cumsum(pdf, -1)
            cdf = torch.cat(
                [torch.zeros_like(cdf[..., :1]), cdf], -1
            )  # (batch, len(bins))

        # Invert CDF
        if (not_converge and total_iters < max_total_iters) or (not model.training):
            u = (
                torch.linspace(0.0, 1.0, steps=N)
                .to(rays_o.device)
                .unsqueeze(0)
                .repeat(cdf.shape[0], 1)
            )
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N]).to(rays_o.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        # Adding samples if we not converged
        if not_converge and total_iters < max_total_iters:
            z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

    z_samples = samples

    near, far = (
        near * torch.ones(rays_d.shape[0], 1).to(rays_o.device),
        far * torch.ones(rays_d.shape[0], 1).to(rays_o.device),
    )

    if N_samples_extra > 0:
        if model.training:
            sampling_idx = torch.randperm(z_vals.shape[1])[:N_samples_extra]
        else:
            sampling_idx = torch.linspace(
                0, z_vals.shape[1] - 1, N_samples_extra
            ).long()
        z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
    else:
        z_vals_extra = torch.cat([near, far], -1)

    z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

    # add some of the near surface points
    idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).to(rays_o.device)
    z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

    return z_vals, z_samples_eik


def get_error_bound(beta, model, sdf, z_vals, dists, d_star):
    density = model.density(sdf.reshape(z_vals.shape), beta=beta)
    shifted_free_energy = torch.cat(
        [torch.zeros(dists.shape[0], 1).to(z_vals.device), dists * density[:, :-1]],
        dim=-1,
    )
    integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
    error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
    error_integral = torch.cumsum(error_per_section, dim=-1)
    bound_opacity = (
        torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
    ) * torch.exp(-integral_estimation[:, :-1])

    return bound_opacity.max(-1)[0]


def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)

    return t_vals, coords


def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def get_sphere_intersections(cam_loc, ray_directions, r=1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(
        ray_directions.view(-1, 1, 3), cam_loc.view(-1, 3, 1)
    ).squeeze(-1)
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r**2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print("BOUNDING SPHERE PROBLEM!")
        exit()

    sphere_intersections = (
        torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    )
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def volume_rendering_volsdf(rgb, density, t_vals, dirs, white_bkgd):
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.ones(t_vals[..., :1].shape, device=t_vals.device)], -1
    )

    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    # LOG SPACE
    free_energy = dists * density.squeeze(-1)
    shifted_free_energy = torch.cat(
        [torch.zeros(t_vals[..., :1].shape, device=t_vals.device), free_energy[:, :-1]],
        dim=-1,
    )  # shift one step
    alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
    transmittance = torch.exp(
        -torch.cumsum(shifted_free_energy, dim=-1)
    )  # probability of everything is empty up to now
    weights = alpha * transmittance  # probability of the ray hits something here

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, weights, depth


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd, nocs=None):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    if nocs is not None:
        comp_nocs = (weights[..., None] * nocs).sum(dim=-2)
        return comp_rgb, acc, weights, comp_nocs
    else:
        return comp_rgb, acc, weights, depth


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):
    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def sample_pdf(bins, weights, origins, directions, t_vals, num_samples, randomized):
    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values
    coords = cast_rays(t_vals, origins, directions)
    return t_vals, coords
