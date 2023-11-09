import torch
from torch import Tensor
from torch.nn.functional import interpolate


def bislerp_rescale(samples: Tensor, tgt_w: int, tgt_h: int) -> Tensor:
    """rescales a batch of images using bislerp algorithm"""

    def slerp(b1: Tensor, b2: Tensor, ratios: Tensor):
        """slerps batches `b1` and `b2` according to `ratios`. batch should be flattened to NxC"""
        ch = b1.shape[-1]

        # get norms and normalize
        b1_norms: Tensor = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms: Tensor = torch.norm(b2, dim=-1, keepdim=True)
        b1_normed: Tensor = b1 / b1_norms
        b2_normed: Tensor = b2 / b2_norms

        # zero out values where norms are zero
        b1_normed[b1_norms.expand(-1, ch) == 0.0] = 0.0
        b2_normed[b2_norms.expand(-1, ch) == 0.0] = 0.0

        # do some slerp
        dot = torch.linalg.vecdot(b1_normed, b2_normed)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # mathematically incorrect but looks nicer
        res = (torch.sin((1.0 - ratios.squeeze(1)) * omega) / so).unsqueeze(1) * b1_normed + (
            (torch.sin(ratios.squeeze(1) * omega) / so).unsqueeze(1) * b2_normed
        )
        res *= (b1_norms * (1.0 - ratios) + b2_norms * ratios).expand(-1, ch)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - ratios) + b2 * ratios)[dot < 1e-5 - 1]

        return res

    def get_bilinear_ratios(old: int, new: int):
        """returns the ratios and indices for bilinear interpolation"""
        coords_1: Tensor = torch.arange(old).reshape((1, 1, 1, -1)).to(torch.float32)
        coords_1 = interpolate(coords_1, size=(1, new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2: Tensor = torch.arange(old).reshape((1, 1, 1, -1)).to(torch.float32) + 1
        coords_2[:, :, :, -1] -= 1
        coords_2 = interpolate(coords_2, size=(1, new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)

        return ratios, coords_1, coords_2

    n, c, h, w = samples.shape

    with torch.device(samples.device):
        # width transform
        ratios, coords_1, coords_2 = get_bilinear_ratios(w, tgt_w)
        coords_1 = coords_1.expand((n, c, h, -1))
        coords_2 = coords_2.expand((n, c, h, -1))
        ratios = ratios.expand((n, 1, h, -1))

        pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
        pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
        ratios = ratios.movedim(1, -1).reshape((-1, 1))

        result = slerp(pass_1, pass_2, ratios)
        result = result.reshape(n, h, tgt_w, c).movedim(-1, 1)

        # height transform
        ratios, coords_1, coords_2 = get_bilinear_ratios(h, tgt_h)
        coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, tgt_w))
        coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, tgt_w))
        ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, tgt_w))

        pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
        pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
        ratios = ratios.movedim(1, -1).reshape((-1, 1))

        result = slerp(pass_1, pass_2, ratios)
        result = result.reshape(n, tgt_h, tgt_w, c).movedim(-1, 1)

    return result.to(samples.device, samples.dtype)
