"""SO(3) rotation augmentation for ACN/SN3D first-order Ambisonics.

Applied on the *waveform* so every downstream feature -- logmel(Y,Z,X), AIV,
intensity, diffuseness, RouteATarget q -- is rotated consistently for free.
For first order the Wigner-D of a rotation R is R itself (conjugated by the
xyz->yzx channel permutation), so a per-sample 3x3 matmul is the exact
sound-field rotation; this holds for SN3D and N3D alike since order-1
normalization is uniform.

W is untouched: the rotation is invisible to the mono semantic encoder, so
each rotated copy is a new spatial sample under *identical* semantic
conditioning -- exactly the augmentation you want for a spatial-only encoder.

Do NOT add time-reversal augmentation alongside this: reversing time turns
decay tails into ramps and corrupts the early/late intensity structure that
the multi-scale diffuseness (DRR-proxy) target is built on.
"""
import math

import torch
from torch import Tensor

# ACN directional channels are (Y, Z, X) = components (y, z, x):
# ch = P @ v with v = (x, y, z).
_P_XYZ_TO_YZX = torch.tensor([[0., 1., 0.],
                              [0., 0., 1.],
                              [1., 0., 0.]])


def random_rotations(batch_size: int, mode: str = "so3",
                     device=None, dtype=torch.float32) -> Tensor:
    """(B, 3, 3) rotation matrices acting on canonical (x, y, z).

    mode='so3' : Haar-uniform over SO(3) (unit quaternions).
    mode='yaw' : uniform rotation about z (up) only -- use if full 3D
                 (sources overhead/underneath) is deemed too aggressive.
    """
    if mode == "so3":
        q = torch.randn(batch_size, 4, device=device, dtype=dtype)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = q.unbind(dim=1)
        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
            2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
        ], dim=1).view(batch_size, 3, 3)
        return R
    if mode == "yaw":
        th = torch.rand(batch_size, device=device, dtype=dtype) * (2 * math.pi)
        c, s = torch.cos(th), torch.sin(th)
        z0, o = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack([c, -s, z0,
                            s,  c, z0,
                            z0, z0, o], dim=1).view(batch_size, 3, 3)
    raise ValueError(f"unknown rotation mode '{mode}'")


def rotate_foa_waveform(audio: Tensor, R: Tensor) -> Tensor:
    """Rotate an ACN FOA waveform batch by R.

    Args:
        audio : (B, 4, S), channels (W, Y, Z, X).
        R     : (B, 3, 3) acting on canonical (x, y, z).
    Returns:
        (B, 4, S).  A plane wave from direction n becomes one from R @ n;
        W is unchanged.
    """
    assert audio.dim() == 3 and audio.size(1) == 4, \
        f"expected (B, 4, S) FOA waveform, got {tuple(audio.shape)}"
    P = _P_XYZ_TO_YZX.to(device=R.device, dtype=R.dtype)
    Mrot = P @ R @ P.transpose(0, 1)                 # (B,3,3) in (y,z,x) basis
    out = audio.clone()
    out[:, 1:] = torch.einsum("bij,bjs->bis",
                              Mrot.to(dtype=audio.dtype), audio[:, 1:])
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S = 64, 4000

    # plane wave from direction n (SN3D): W = s(t), (Y,Z,X) = (ny, nz, nx) s(t)
    n = torch.randn(B, 3)
    n = n / n.norm(dim=1, keepdim=True)
    sig = torch.randn(B, 1, S)
    audio = torch.cat([sig,
                       n[:, 1:2, None] * sig,       # Y = y * s
                       n[:, 2:3, None] * sig,       # Z = z * s
                       n[:, 0:1, None] * sig],      # X = x * s
                      dim=1)

    for mode in ("so3", "yaw"):
        R = random_rotations(B, mode)
        # orthonormality / det +1
        I3 = torch.eye(3).expand(B, 3, 3)
        assert torch.allclose(R @ R.transpose(1, 2), I3, atol=1e-5)
        assert torch.allclose(torch.linalg.det(R), torch.ones(B), atol=1e-5)

        rot = rotate_foa_waveform(audio, R)
        assert torch.equal(rot[:, 0], audio[:, 0]), "W must be untouched"

        # recover direction from rotated channels: v = (X, Y, Z)/W
        n_rot = torch.stack([rot[:, 3, 0], rot[:, 1, 0], rot[:, 2, 0]],
                            dim=1) / sig[:, 0, 0:1]
        n_expected = torch.einsum("bij,bj->bi", R, n)
        assert torch.allclose(n_rot, n_expected, atol=1e-4), mode
        print(f"{mode}: plane wave n -> R@n exactly, W invariant: OK")

    # Haar-uniformity sanity: mean of rotated fixed vector ~ 0 over many draws
    R = random_rotations(20000, "so3")
    v = torch.einsum("bij,j->bi", R, torch.tensor([1., 0., 0.]))
    assert v.mean(dim=0).abs().max() < 0.02
    print("SO(3) sampling is (statistically) uniform: OK")
    print("all foa_rotation tests passed")