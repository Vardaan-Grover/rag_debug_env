"""
tests/test_fault_math.py
------------------------
Tests for server/fault_math.py.

Verifies that:
- Each fault type degrades retrieval relative to a no-fault baseline.
- apply_faults() with an empty fault set returns the original matrix (up to float precision).
- make_noise() returns the expected keys.
- All output values stay in [0, 1] after clipping.
"""

import numpy as np
import pytest

from server.fault_math import apply_faults, make_noise
from models import FaultType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_QUERIES = 8
N_CHUNKS = 40
SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def S_true(rng):
    """A realistic S_true matrix: relevant chunks score ~0.7-0.9, others ~0.2-0.5."""
    S = rng.uniform(0.2, 0.5, (N_QUERIES, N_CHUNKS)).astype(np.float32)
    # Spike the first 2 chunks of each query as "relevant" with high scores.
    for i in range(N_QUERIES):
        S[i, i % N_CHUNKS] = rng.uniform(0.75, 0.92)
        S[i, (i + 1) % N_CHUNKS] = rng.uniform(0.70, 0.88)
    return S


@pytest.fixture
def noise(rng):
    return make_noise(rng, (N_QUERIES, N_CHUNKS))


@pytest.fixture
def dupe_ids(rng):
    return rng.choice(N_CHUNKS, size=max(1, N_CHUNKS // 7), replace=False)


def _apply(S, fault_types, noise, dupe_ids, **kwargs):
    defaults = dict(
        config_chunk_size=512,
        config_context_limit=4096,
        config_use_reranking=False,
        config_chunk_overlap=50,
    )
    defaults.update(kwargs)
    return apply_faults(
        S=S,
        fault_types=set(fault_types),
        noise=noise,
        dupe_ids=dupe_ids,
        **defaults,
    )


# ---------------------------------------------------------------------------
# Basic contracts
# ---------------------------------------------------------------------------

def test_no_faults_returns_original(S_true, noise, dupe_ids):
    """With no faults, the matrix should be unchanged (modulo float32 copy)."""
    result = _apply(S_true, [], noise, dupe_ids)
    np.testing.assert_array_almost_equal(result, S_true, decimal=5)


def test_output_clipped_to_unit_interval(S_true, noise, dupe_ids):
    """All faults combined must still produce values in [0, 1]."""
    all_faults = [
        FaultType.CHUNK_TOO_LARGE,
        FaultType.CHUNK_TOO_SMALL,
        FaultType.THRESHOLD_TOO_LOW,
        FaultType.THRESHOLD_TOO_HIGH,
        FaultType.TOP_K_TOO_SMALL,
        FaultType.DUPLICATE_FLOODING,
        FaultType.CONTEXT_OVERFLOW,
        FaultType.NO_RERANKING,
    ]
    result = _apply(S_true, all_faults, noise, dupe_ids)
    assert result.min() >= 0.0, f"Minimum value {result.min()} < 0"
    assert result.max() <= 1.0, f"Maximum value {result.max()} > 1"


def test_make_noise_returns_expected_keys(rng):
    shape = (5, 20)
    n = make_noise(rng, shape)
    assert FaultType.CHUNK_TOO_SMALL in n
    assert FaultType.THRESHOLD_TOO_LOW in n
    assert FaultType.NO_RERANKING in n
    for v in n.values():
        assert v.shape == shape
        assert v.dtype == np.float32


def test_apply_faults_does_not_mutate_input(S_true, noise, dupe_ids):
    original = S_true.copy()
    _apply(S_true, [FaultType.CHUNK_TOO_LARGE, FaultType.THRESHOLD_TOO_HIGH], noise, dupe_ids)
    np.testing.assert_array_equal(S_true, original)


# ---------------------------------------------------------------------------
# Each fault degrades in the expected direction
# ---------------------------------------------------------------------------

def _mean_top1_score(S: np.ndarray) -> float:
    """Mean of the max score per query — proxy for how retrieval-friendly the matrix is."""
    return float(S.max(axis=1).mean())


def _score_std(S: np.ndarray) -> float:
    return float(S.std())


def test_chunk_too_large_smears_scores(S_true, noise, dupe_ids):
    """CHUNK_TOO_LARGE applies a box filter that blurs score peaks downward."""
    result = _apply(S_true, [FaultType.CHUNK_TOO_LARGE], noise, dupe_ids, config_chunk_size=2048)
    # Box filter reduces the peak scores
    assert _mean_top1_score(result) < _mean_top1_score(S_true), (
        "CHUNK_TOO_LARGE should smear peak scores downward"
    )


def test_threshold_too_high_deflates_scores(S_true, noise, dupe_ids):
    """THRESHOLD_TOO_HIGH multiplies all scores by 0.55, reducing absolute values."""
    result = _apply(S_true, [FaultType.THRESHOLD_TOO_HIGH], noise, dupe_ids)
    assert result.mean() < S_true.mean() * 0.65, (
        "THRESHOLD_TOO_HIGH should significantly deflate scores"
    )


def test_top_k_too_small_compresses_score_range(S_true, noise, dupe_ids):
    """TOP_K_TOO_SMALL compresses scores toward 0.5, reducing std."""
    result = _apply(S_true, [FaultType.TOP_K_TOO_SMALL], noise, dupe_ids)
    assert _score_std(result) < _score_std(S_true), (
        "TOP_K_TOO_SMALL should compress score variance"
    )


def test_top_k_too_small_less_severe_with_reranking(S_true, noise, dupe_ids):
    """Enabling reranking should reduce TOP_K_TOO_SMALL severity."""
    without_rerank = _apply(S_true, [FaultType.TOP_K_TOO_SMALL], noise, dupe_ids,
                            config_use_reranking=False)
    with_rerank = _apply(S_true, [FaultType.TOP_K_TOO_SMALL], noise, dupe_ids,
                         config_use_reranking=True)
    # With reranking, score variance should be closer to original (less compressed)
    std_without = _score_std(without_rerank)
    std_with = _score_std(with_rerank)
    assert std_with > std_without, (
        "Reranking should partially restore score spread under TOP_K_TOO_SMALL"
    )


def test_duplicate_flooding_boosts_dupe_columns(S_true, noise, dupe_ids):
    """DUPLICATE_FLOODING boosts duplicate chunk columns."""
    result = _apply(S_true, [FaultType.DUPLICATE_FLOODING], noise, dupe_ids)
    # Mean score of duplicate columns should be higher after flooding
    assert result[:, dupe_ids].mean() > S_true[:, dupe_ids].mean(), (
        "DUPLICATE_FLOODING should boost duplicate chunk scores"
    )


def test_duplicate_flooding_reduced_with_reranking(S_true, noise, dupe_ids):
    """Reranking should reduce the duplicate flooding boost."""
    without = _apply(S_true, [FaultType.DUPLICATE_FLOODING], noise, dupe_ids,
                     config_use_reranking=False)
    with_rr = _apply(S_true, [FaultType.DUPLICATE_FLOODING], noise, dupe_ids,
                     config_use_reranking=True)
    assert with_rr[:, dupe_ids].mean() < without[:, dupe_ids].mean(), (
        "Reranking should reduce duplicate flooding boost"
    )


def test_context_overflow_zeroes_tail_columns(S_true, noise, dupe_ids):
    """CONTEXT_OVERFLOW zeroes chunks beyond the context window cutoff."""
    tight_limit = 512  # Very small → cuts most columns
    result = _apply(S_true, [FaultType.CONTEXT_OVERFLOW], noise, dupe_ids,
                    config_context_limit=tight_limit)
    cutoff = max(1, int(N_CHUNKS * tight_limit / 16384))
    if cutoff < N_CHUNKS:
        assert result[:, cutoff:].sum() == 0.0, (
            "Columns beyond context cutoff should be zeroed"
        )


def test_no_reranking_adds_noise(S_true, noise, dupe_ids):
    """NO_RERANKING fault adds noise when reranking is off."""
    result_off = _apply(S_true, [FaultType.NO_RERANKING], noise, dupe_ids,
                        config_use_reranking=False)
    result_on = _apply(S_true, [FaultType.NO_RERANKING], noise, dupe_ids,
                       config_use_reranking=True)
    # When reranking is on, NO_RERANKING fault is suppressed → output is closer to S_true
    diff_off = float(np.abs(result_off - S_true).mean())
    diff_on = float(np.abs(result_on - S_true).mean())
    assert diff_off > diff_on, (
        "NO_RERANKING should only add noise when reranking is disabled"
    )


def test_chunk_too_small_noise_reduced_by_overlap(S_true, noise, dupe_ids):
    """Higher chunk_overlap reduces CHUNK_TOO_SMALL noise sigma."""
    low_overlap = _apply(S_true, [FaultType.CHUNK_TOO_SMALL], noise, dupe_ids,
                         config_chunk_size=128, config_chunk_overlap=0)
    high_overlap = _apply(S_true, [FaultType.CHUNK_TOO_SMALL], noise, dupe_ids,
                          config_chunk_size=128, config_chunk_overlap=450)
    diff_low = float(np.abs(low_overlap - S_true).mean())
    diff_high = float(np.abs(high_overlap - S_true).mean())
    assert diff_low > diff_high, (
        "Higher overlap should reduce CHUNK_TOO_SMALL noise impact"
    )
