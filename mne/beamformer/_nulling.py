"""Compute Nulling Beamformer."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.meas_info import _simplify_info
from .._fiff.pick import pick_channels_cov, pick_info
from ..forward import _subject_from_forward
from ..minimum_norm.inverse import _check_depth, _check_reference, combine_xyz
from ..rank import compute_rank
from ..source_estimate import _get_src_type, _make_stc
from ..utils import (
    _check_channels_spatial_filter,
    _check_info_inv,
    _check_one_ch_type,
    logger,
    verbose,
)
from ._compute_beamformer import (
    Beamformer,
    _check_src_type,
    _compute_beamformer,
    _compute_nulling_beamformer,
    _compute_power,
    _prepare_beamformer_input,
    _proj_whiten_data,
)


def make_nulling_beamformer(
    info,
    forward,
    data_cov,
    null_label,
    target_label,
    noise_cov=None,
    pick_ori=None,
    inversion="single", # TODO: this is only for now! 
    rank="info",
    label=None,
    depth=None,
):
    """Compute nulling beamformer spatial filter.
    
    The nulling beamformer suppresses activity from specified source
    locations while maintaining sensitivity to other sources.
    
    Parameters
    ----------
    null_label : Label
                Label of source locations to be nulled.

    Returns
    -------
    

    
    Notes
    -----
    The nulling beamformer constrains the spatial filters to have
    zero response at specified source locations [1]_.
    The original reference is :footcite:``.

    To obtain the Sekihara unit-noise-gain vector beamformer, you should use
    ``weight_norm='unit-noise-gain', pick_ori='vector'`` followed by
    :meth:`vec_stc.project('pca', src) <mne.VectorSourceEstimate.project>`.

    .. versionchanged:: 0.21
       The computations were extensively reworked, and the default for
       ``weight_norm`` was set to ``'unit-noise-gain-invariant'``.

    References
    ----------
    .. footbibliography::
    """

    # check number of sensor types present in the data and ensure a noise cov
    info = _simplify_info(info, keep=("proc_history",))
    noise_cov, _, allow_mismatch = _check_one_ch_type(
        "lcmv", info, forward, data_cov, noise_cov
    )
    
    # XXX we need this extra picking step (can't just rely on minimum norm's
    # because there can be a mismatch. Should probably add an extra arg to
    # _prepare_beamformer_input at some point (later)
    picks = _check_info_inv(info, forward, data_cov, noise_cov)
    info = pick_info(info, picks)
    data_rank = compute_rank(data_cov, rank=rank, info=info)
    noise_rank = compute_rank(noise_cov, rank=rank, info=info)
    for key in data_rank:
        if (
            key not in noise_rank or data_rank[key] != noise_rank[key]
        ) and not allow_mismatch:
            raise ValueError(
                f"{key} data rank ({data_rank[key]}) did not match the noise rank ("
                f"{noise_rank.get(key, None)})"
            )
    del noise_rank
    rank = data_rank
    logger.info(f"Making LCMV beamformer with rank {rank}")
    del data_rank
    depth = _check_depth(depth, "depth_sparse")
    if inversion == "single":
        depth["combine_xyz"] = False

    (
        is_free_ori,
        info,
        proj,
        vertno,
        G,
        whitener,
        nn,
        orient_std,
    ) = _prepare_beamformer_input(
        info,
        forward,
        label,
        pick_ori,
        noise_cov=noise_cov,
        rank=rank,
        pca=False,
        **depth,
    )

    # obtain cata covariance from channels in info
    ch_names = list(info["ch_names"])
    data_cov = pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov._get_square()

    # determine number of orientations
    n_orient = 3 if is_free_ori else 1

    # TODO: for now only one target source
    # get the vertex id of the target
    target_idx = None
    src = forward['src']
    for hemi_idx, hemi in enumerate(src):
        hemi_name = 'lh' if hemi_idx == 0 else 'rh'
        if target_label.hemi != hemi_name:
            continue
        # get the vertex numbers in the label
        label_vertex = target_label.get_vertices_used(vertices=src[hemi_idx]['vertno'])[0]
        # get the vertex numbers in the source space
        src_verts = hemi["vertno"]
        # find the index of the target label vertex in the source space
        mask = np.isin(src_verts, label_vertex)
        target_idx = np.where(mask)[0]

    # get the null indices 
    null_idx = _get_label_idxs(null_label, forward['src'])
    
    # compute rank
    rank_int = sum(rank.values())
    del rank
    
    # compute beamformer weights with nulling constraints
    W, max_power_ori = _compute_nulling_beamformer(
        G,
        Cm,
        null_idx,
        target_idx,
        pick_ori,
        n_orient,
    )

    # get src type to store with filters for _make_stc
    src_type = _get_src_type(forward["src"], vertno)

    # get subject to store with filters
    subject_from = _subject_from_forward(forward)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, "vector"] else False
    is_ssp = bool(info["projs"])

    filters = Beamformer(
        kind="LCMV",
        weights=W,
        data_cov=data_cov,
        noise_cov=noise_cov,
        whitener=whitener,
        weight_norm=None,
        pick_ori=None,
        ch_names=ch_names,
        proj=proj,
        is_ssp=is_ssp,
        vertices=vertno,
        is_free_ori=is_free_ori,
        n_sources=forward["nsource"],
        src_type=src_type,
        source_nn=forward["source_nn"].copy(),
        subject=subject_from,
        rank=rank_int,
        max_power_ori=max_power_ori,
        inversion=inversion,
    )
    
    return filters

def _get_label_idxs(null_label, src):
    """Get the indices of the vertices in the null label."""
    
    null_idxs = []
    for hemi_idx, hemi in enumerate(src):
        hemi_name = 'lh' if hemi_idx == 0 else 'rh'
        if null_label.hemi != hemi_name:
            continue
        # get the vertex numbers in the label
        label_verts = null_label.get_vertices_used()
        # get the vertex numbers in the source space
        src_verts = hemi["vertno"]
        # find the indices of the label vertices in the source space
        mask = np.isin(src_verts, label_verts)
        null_idxs = np.where(mask)[0]

    return null_idxs