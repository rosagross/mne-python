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
    _compute_power,
    _prepare_beamformer_input,
    _proj_whiten_data,
)


def make_nulling_beamformer(
    info,
    forward,
    data_cov,
    null_sources,  # NEW: sources to null
    null_orientations=None,  # NEW: orientations to null
    reg=0.05,
    noise_cov=None,
    pick_ori=None,
    rank="info",
    weight_norm="unit-noise-gain-invariant",
    reduce_rank=False,
    depth=None,
    inversion="matrix",
    verbose=None,
):
    """Compute nulling beamformer spatial filter.
    
    The nulling beamformer suppresses activity from specified source
    locations while maintaining sensitivity to other sources.
    
    Parameters
    ----------
    null_sources : array, shape (n_null, 3) or SourceSpaces
        Source locations to suppress. Can be vertex indices or 
        coordinates in head space.
    null_orientations : array, shape (n_null, 3) | None
        Orientations of sources to null. If None, uses surface normals
        from forward model.
    ... (other parameters same as make_lcmv)
    
    Returns
    -------
    filters : instance of Beamformer
        Dictionary containing filter weights from nulling beamformer.
        Additional keys compared to LCMV:
            'null_sources' : array
                Source locations that were nulled.
            'null_orientations' : array
                Orientations that were nulled.
    
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
    # Reuse LCMV preprocessing
    # ... (similar setup code)
    
    # Compute nulling constraints
    null_leadfield = _compute_null_leadfield(
        forward, null_sources, null_orientations
    )
    
    # Compute nulling beamformer weights
    W = _compute_nulling_weights(
        G, Cm, null_leadfield, reg, weight_norm, ...
    )
    
    # Return modified Beamformer object
    filters = Beamformer(
        kind="nulling",  # Different kind
        weights=W,
        null_sources=null_sources,  # Additional info
        null_orientations=null_orientations,
        # ... rest same as LCMV
    )
    
    return filters