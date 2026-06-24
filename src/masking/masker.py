from typing import Tuple, Optional
from torch import nn
import torch

from .utils import generate_masks_batch


class SpatialMaskMaker(nn.Module):
    """Generates visible-patch index tensors for EmbisonicsMAE.

    Wraps ``generate_masks_batch`` which returns a boolean mask where
    ``True = masked`` (hidden from the encoder).  This class inverts that
    convention and returns the **indices of the visible (unmasked) patches**
    as expected by ``EmbisonicsMAE._prepare_batch``:

        context_idx : (B, L_vis)   long tensor of visible patch indices

    Args:
        mask_patch:
            Number of patches to mask (hide from the encoder).
        context_cluster:
            If True, masks are generated in spatial clusters; otherwise
            patches are sampled uniformly at random.
    """

    def __init__(
        self,
        mask_patch: int = 10,
        context_cluster: bool = True,
    ):
        super().__init__()
        self.mask_patch = mask_patch
        self.context_cluster = context_cluster

    def forward(
        self,
        local_features: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        n_patches: Optional[int] = None,
    ) -> torch.Tensor:
        """Return visible-patch indices suitable for EmbisonicsMAE.

        Parameters
        ----------
        local_features:
            If provided, ``batch_size`` and ``n_patches`` are inferred from
            its shape ``(B, P, D)``.  Pass ``None`` and supply ``batch_size``
            / ``n_patches`` explicitly when calling outside a forward pass.
        batch_size:
            Number of samples in the batch.  Ignored when ``local_features``
            is not None.
        n_patches:
            Total number of patches in the sequence (``num_patches`` on the
            model).  Ignored when ``local_features`` is not None.

        Returns
        -------
        context_idx : (B, L_vis) LongTensor
            Indices of the **visible** (unmasked) patches for each sample.
            ``L_vis = n_patches - mask_patch``.
        """
        if local_features is not None:
            batch_size, n_patches, _ = local_features.shape

        # bool mask: True = masked/hidden, shape (B, n_patches)
        masked_bool = generate_masks_batch(
            B=batch_size,
            sequence_len=n_patches,
            mask_patch=self.mask_patch,
            cluster_ctx=self.context_cluster,
        )

        # Invert: True = visible.  Collect column indices per row.
        visible_bool = ~masked_bool                           # (B, n_patches)
        # nonzero gives [[row, col], ...]; we want (B, L_vis)
        context_idx = visible_bool.nonzero(as_tuple=False)   # (B*L_vis, 2)
        L_vis = n_patches - self.mask_patch
        context_idx = context_idx[:, 1].reshape(batch_size, L_vis)  # (B, L_vis)

        return context_idx