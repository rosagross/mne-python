# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_less

import mne
from mne import read_label
from mne.beamformer import Beamformer, apply_nulling, make_nulling_beamformer
from mne.beamformer.tests.test_lcmv import _get_data
from mne.datasets import testing
from mne.utils import catch_logging

data_path = testing.data_path(download=False)
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis_trunc-meg-eeg-oct-4-fwd.fif"
fname_label = data_path / "MEG" / "sample" / "labels" / "Aud-lh.label"


@testing.requires_testing_data
@pytest.mark.parametrize("weight_norm", [None, "unit-noise-gain"])
def test_make_nulling_beamformer_basic(weight_norm):
    """Test basic nulling beamformer computation."""
    (
        raw,
        epochs,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
        weight_norm=weight_norm,
    )

    assert isinstance(filters, Beamformer)
    assert "weights" in filters
    assert filters["weights"].shape[1] == len(evoked.ch_names)

    stc = apply_nulling(evoked, filters)
    assert stc is not None


@testing.requires_testing_data
@pytest.mark.parametrize("null_reduction", [None, "none", "auto", 10, 0.6])
def test_nulling_beamformer_null_reduction(null_reduction):
    """Test nulling beamformer with different null_reduction settings."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
        null_reduction=null_reduction,
        weight_norm=None,
    )

    assert isinstance(filters, Beamformer)
    stc = apply_nulling(evoked, filters)
    assert stc is not None


@testing.requires_testing_data
@pytest.mark.parametrize("inversion", ["matrix"])
def test_nulling_beamformer_inversion(inversion):
    """Test nulling beamformer with different inversion settings."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
        inversion=inversion,
        weight_norm=None,
    )

    assert isinstance(filters, Beamformer)
    stc = apply_nulling(evoked, filters)
    assert stc is not None


@testing.requires_testing_data
def test_nulling_constraints_enforced():
    """Test that nulling beamformer actually nulls the specified sources."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
        weight_norm=None,
    )

    from mne.beamformer._nulling import _get_label_idxs

    null_idxs = _get_label_idxs(null_label, forward["src"])

    G = forward["sol"]["data"]
    n_channels = G.shape[0]
    n_sources = G.shape[1] // 3
    W = filters["weights"]

    W_reshaped = W.reshape(n_sources, 3, n_channels)

    for null_idx in null_idxs[:5]:
        for ori in range(3):
            gain = W_reshaped[null_idx, ori, :] @ G[:, null_idx * 3 + ori]
            assert_allclose(gain, 0.0, atol=1e-5)


@testing.requires_testing_data
def test_nulling_beamformer_volume_source_space():
    """Test nulling beamformer with volume source space."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        _,
        _,
        _,
        forward_vol,
    ) = _get_data(epochs=True, proj=True)

    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.0), head_radius=0.080)
    src = mne.setup_volume_source_space(
        pos=25.0, sphere=sphere, mindist=5.0, exclude=2.0
    )
    fwd_sphere = mne.make_forward_solution(
        evoked.info, None, src, sphere
    )

    label = mne.read_label(fname_label)
    src_verts = src[0]["vertno"]
    null_vertices = np.intersect1d(label.vertices, src_verts)

    if len(null_vertices) == 0:
        pytest.skip("No vertices in volume source space overlap with label")

    null_label = label.copy()
    null_label.vertices = [null_vertices[0]]
    null_label.hemi = "lh"

    filters = make_nulling_beamformer(
        evoked.info,
        fwd_sphere,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
        weight_norm=None,
    )

    assert isinstance(filters, Beamformer)
    stc = apply_nulling(evoked, filters)
    assert stc is not None


@testing.requires_testing_data
def test_nulling_beamformer_reg_parameter():
    """Test nulling beamformer with different regularization values."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    for reg in [0.0, 0.01, 0.1]:
        filters = make_nulling_beamformer(
            evoked.info,
            forward,
            data_cov,
            null_label=null_label,
            noise_cov=noise_cov,
            reg=reg,
            weight_norm=None,
        )
        assert isinstance(filters, Beamformer)
        stc = apply_nulling(evoked, filters)
        assert stc is not None


@testing.requires_testing_data
def test_nulling_beamformer_logging(tmp_path):
    """Test that nulling beamformer produces expected log output."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    with catch_logging() as log:
        filters = make_nulling_beamformer(
            evoked.info,
            forward,
            data_cov,
            null_label=null_label,
            noise_cov=noise_cov,
            reg=0.05,
            verbose=True,
        )
    log_text = log.getvalue()
    assert "beamformer" in log_text.lower()
    assert filters is not None


@testing.requires_testing_data
def test_nulling_beamformer_repr():
    """Test the __repr__ method of the beamformer filters."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
    )

    repr_str = repr(filters)
    assert "Beamformer" in repr_str
    assert "vert" in repr_str
    assert "ch" in repr_str


@testing.requires_testing_data
def test_nulling_beamformer_no_null_overlap():
    """Test nulling beamformer when null label doesn't overlap with source space."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    label = mne.Label(
        vertices=np.array([999999]),
        pos=np.array([[0.0, 0.0, 0.0]]),
        normals=np.array([[0.0, 0.0, 1.0]]),
        values=np.array([1.0]),
        hemi="lh",
        subject="sample",
    )

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=label,
        noise_cov=noise_cov,
        reg=0.05,
        weight_norm=None,
    )

    assert isinstance(filters, Beamformer)
    stc = apply_nulling(evoked, filters)
    assert stc is not None


@testing.requires_testing_data
def test_nulling_beamformer_comparison_with_lcmv():
    """Test that nulling beamformer without null is similar to LCMV."""
    from mne.beamformer import make_lcmv

    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    label = mne.Label(
        vertices=np.array([999999]),
        pos=np.array([[0.0, 0.0, 0.0]]),
        normals=np.array([[0.0, 0.0, 1.0]]),
        values=np.array([1.0]),
        hemi="lh",
        subject="sample",
    )

    filters_lcmv = make_lcmv(
        evoked.info,
        forward,
        data_cov,
        noise_cov=noise_cov,
        reg=0.05,
        weight_norm=None,
    )

    filters_nulling = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=label,
        noise_cov=noise_cov,
        reg=0.05,
        weight_norm=None,
    )

    stc_lcmv = mne.beamformer.apply_lcmv(evoked, filters_lcmv)
    stc_nulling = apply_nulling(evoked, filters_nulling)

    assert_allclose(stc_lcmv.data, stc_nulling.data, rtol=1e-10)


@testing.requires_testing_data
def test_apply_nulling_evoked_types():
    """Test apply_nulling returns correct source estimate types."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
    )

    stc = apply_nulling(evoked, filters)

    assert isinstance(stc, mne.SourceEstimate)
    assert stc.data.shape[0] == forward["nsource"]


@testing.requires_testing_data
def test_nulling_beamformer_copy():
    """Test that beamformer filters can be copied."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        _,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = read_label(fname_label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
    )

    filters_copy = filters.copy()

    assert filters is not filters_copy
    assert_array_almost_equal(
        filters["weights"],
        filters_copy["weights"],
    )

# TODO: implement test for different weight norms
# TODO: implement test for different pick_ori options 
# TODO: implement test for combination of pick ori and weight norm


