# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
)

import mne
from mne import read_label
from mne.beamformer import (
    Beamformer,
    apply_nulling,
    make_nulling_beamformer,
    read_beamformer,
)
from mne.beamformer.tests.test_lcmv import _get_data
from mne.datasets import testing
from mne.utils import catch_logging, object_diff

data_path = testing.data_path(download=False)
fname_label = data_path / "MEG" / "sample" / "labels" / "Aud-lh.label"


def _get_null_label_for_fwd(forward, label):
    if forward["src"][0]["type"] == "vol":
        src_verts = forward["src"][0]["vertno"]
        null_vertices = np.intersect1d(label.vertices, src_verts)
        if len(null_vertices) == 0:
            pytest.skip("No vertices in volume source space overlap with label")
        null_label = label.copy()
        null_label.vertices = [null_vertices[0]]
        null_label.hemi = "lh"
        return null_label
    return label


@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize(
    "reg, proj, kind",
    [
        (0.01, True, "volume"),
        (0.0, False, "volume"),
        (0.01, False, "surface"),
        (0.0, True, "surface"),
    ],
)
def test_make_nulling_bem(tmp_path, reg, proj, kind):
    """Test nulling beamformer with evoked data and I/O."""
    pytest.importorskip("h5io")
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        forward_vol,
    ) = _get_data(proj=proj)

    if kind == "surface":
        fwd = forward
    else:
        fwd = forward_vol
        assert kind == "volume"

    null_label = _get_null_label_for_fwd(fwd, label)

    filters = make_nulling_beamformer(
        evoked.info, fwd, data_cov, null_label=null_label, reg=reg, noise_cov=noise_cov
    )
    stc = apply_nulling(evoked, filters)
    stc.crop(0.02, None)

    stc_pow = np.sum(np.abs(stc.data), axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert 0.08 < tmax < 0.15, tmax
    assert 0.6 < np.max(max_stc) < 4.0, np.max(max_stc)

    # Smoke test for label= support for surfaces only
    ctx = nullcontext() if kind == "surface" else pytest.raises(
        ValueError, match="volume source space"
    )
    with ctx:
        make_nulling_beamformer(
            evoked.info,
            fwd,
            data_cov,
            null_label=null_label,
            reg=reg,
            noise_cov=noise_cov,
            label=label,
        )

    # Test if spatial filter contains src_type
    assert filters["src_type"] == kind

    # __repr__
    assert len(evoked.ch_names) == 22
    assert len(evoked.info["projs"]) == (3 if proj else 0)
    assert len(evoked.info["bads"]) == 2
    rank = 17 if proj else 20
    assert "LCMV" in repr(filters)
    assert "unknown subject" not in repr(filters)
    assert f"{fwd['nsource']} vert" in repr(filters)
    assert "20 ch" in repr(filters)
    assert f"rank {rank}" in repr(filters)

    # I/O
    fname = tmp_path / "filters.h5"
    with pytest.warns(RuntimeWarning, match="-lcmv.h5"):
        filters.save(fname)
    filters_read = read_beamformer(fname)
    assert isinstance(filters, Beamformer)
    assert isinstance(filters_read, Beamformer)
    filters_read["rank"] = int(filters_read["rank"])
    filters["rank"] = int(filters["rank"])
    assert object_diff(filters, filters_read) == ""


@testing.requires_testing_data
def test_apply_nulling_channel_selection_and_src_type():
    """Test channel selection and src_type warnings."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

    # Apply filter made with a reduced channel set to full data
    evoked_ch = deepcopy(evoked)
    evoked_ch.pick(evoked_ch.ch_names[1:])
    filters = make_nulling_beamformer(
        evoked_ch.info,
        forward,
        data_cov,
        null_label=null_label,
        reg=0.01,
        noise_cov=noise_cov,
    )
    stc = apply_nulling(evoked, filters)
    stc_ch = apply_nulling(evoked_ch, filters)
    assert_array_almost_equal(stc.data, stc_ch.data)

    # check whether a filters object without src_type throws expected warning
    del filters["src_type"]
    with pytest.warns(RuntimeWarning, match="spatial filter does not contain src_type"):
        apply_nulling(evoked, filters)


@testing.requires_testing_data
@pytest.mark.parametrize("weight_norm", [None, "unit-noise-gain"])
def test_make_nulling_beamformer_basic(weight_norm):
    """Test basic nulling beamformer computation."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

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
def test_make_nulling_beamformer_bad_pick_ori():
    """Test invalid pick_ori handling."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

    with pytest.raises(ValueError, match="pick_ori"):
        make_nulling_beamformer(
            evoked.info,
            forward,
            data_cov,
            null_label=null_label,
            noise_cov=noise_cov,
            pick_ori="bad",
        )


@testing.requires_testing_data
def test_make_nulling_beamformer_unigain_invariant_not_implemented():
    """Test that unit-noise-gain-invariant raises NotImplementedError."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

    with pytest.raises(NotImplementedError, match="unit-noise-gain-invariant"):
        make_nulling_beamformer(
            evoked.info,
            forward,
            data_cov,
            null_label=null_label,
            noise_cov=noise_cov,
            weight_norm="unit-noise-gain-invariant",
        )


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

    null_label = _get_null_label_for_fwd(forward, read_label(fname_label))

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
@pytest.mark.parametrize("null_reduction", [None, "none", "auto", 10, 0.6])
def test_nulling_beamformer_null_reduction(null_reduction):
    """Test nulling beamformer with different null_reduction settings."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

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
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

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
def test_nulling_beamformer_reg_parameter():
    """Test nulling beamformer with different regularization values."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

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
def test_nulling_beamformer_logging():
    """Test that nulling beamformer produces expected log output."""
    (
        _,
        _,
        evoked,
        data_cov,
        noise_cov,
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

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
        label,
        forward,
        _,
        _,
        _,
    ) = _get_data(epochs=True, proj=True)

    null_label = _get_null_label_for_fwd(forward, label)

    filters = make_nulling_beamformer(
        evoked.info,
        forward,
        data_cov,
        null_label=null_label,
        noise_cov=noise_cov,
        reg=0.05,
        pick_ori="vector",
    )

    stc = apply_nulling(evoked, filters)

    assert isinstance(stc, mne.VectorSourceEstimate)
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

    null_label = _get_null_label_for_fwd(forward, read_label(fname_label))

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
