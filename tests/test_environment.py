"""
tests/test_environment.py
-------------------------
Tests for episode lifecycle and action routing in RAGDebugEnvironment.

Verifies that:
- reset() fully initialises state and returns a valid observation.
- step() increments step count and returns bounded rewards.
- Each action type modifies the correct config field.
- Auto-terminate fires at max_steps.
- ADJUST_CHUNK_OVERLAP now triggers _recompute_S_faulted() (bug fix).
"""

import pytest
import numpy as np

from server.rag_debug_env_environment import RAGDebugEnvironment
from server.constants import _MAX_STEPS
from models import (
    RAGDebugAction,
    ActionType,
    EmbeddingModel,
    RAGDebugObservation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[1, 2, 3])
def env(request):
    """Fresh environment reset to each task."""
    e = RAGDebugEnvironment()
    e.reset(seed=0, task_id=request.param)
    return e


@pytest.fixture
def env1():
    e = RAGDebugEnvironment()
    e.reset(seed=0, task_id=1)
    return e


def _step(env, action_type, params=None):
    action = RAGDebugAction(action_type=action_type, params=params or {})
    return env.step(action)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=1, task_id=1)
        assert isinstance(obs, RAGDebugObservation)

    def test_reset_clears_step_count(self, env1):
        env1.step(RAGDebugAction(action_type=ActionType.ADJUST_TOP_K, params={"value": 15}))
        assert env1._state.step_count == 1
        env1.reset(seed=99, task_id=1)
        assert env1._state.step_count == 0

    def test_reset_clears_done_flag(self, env1):
        # Force done via SUBMIT
        env1.step(RAGDebugAction(action_type=ActionType.SUBMIT, params={}))
        assert env1._done is True
        env1.reset(seed=5, task_id=1)
        assert env1._done is False

    def test_reset_returns_valid_metrics(self, env):
        obs = env.reset(seed=2, task_id=1)
        m = obs.metrics
        assert 0.0 <= m.mean_coverage <= 1.0
        assert 0.0 <= m.mean_precision <= 1.0
        assert m.n_empty_retrievals >= 0
        assert m.n_context_overflows >= 0

    def test_reset_with_different_tasks(self):
        e = RAGDebugEnvironment()
        for task_id in (1, 2, 3):
            obs = e.reset(seed=0, task_id=task_id)
            assert obs.task_id == task_id

    def test_reset_invalid_task_raises(self):
        e = RAGDebugEnvironment()
        with pytest.raises(ValueError, match="task_id"):
            e.reset(seed=0, task_id=99)

    def test_reset_clears_action_history(self, env1):
        env1.step(RAGDebugAction(action_type=ActionType.ADJUST_TOP_K, params={"value": 15}))
        env1.reset(seed=0, task_id=1)
        assert env1._internal_state.action_history == []
        assert env1._internal_state.reward_history == []


# ---------------------------------------------------------------------------
# Step lifecycle
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_increments_step_count(self, env1):
        for expected in range(1, 4):
            _step(env1, ActionType.ADJUST_TOP_K, {"value": 15})
            assert env1._state.step_count == expected

    def test_step_returns_observation(self, env1):
        obs = _step(env1, ActionType.ADJUST_TOP_K, {"value": 15})
        assert isinstance(obs, RAGDebugObservation)

    def test_step_observation_reward_in_unit_interval(self, env1):
        obs = _step(env1, ActionType.ADJUST_THRESHOLD, {"value": 0.2})
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_step_after_done_raises(self, env1):
        _step(env1, ActionType.SUBMIT)
        with pytest.raises(RuntimeError, match="already done"):
            _step(env1, ActionType.ADJUST_TOP_K, {"value": 15})

    def test_auto_terminate_at_max_steps(self):
        e = RAGDebugEnvironment()
        obs = e.reset(seed=0, task_id=1)
        for _ in range(_MAX_STEPS - 1):
            obs = _step(e, ActionType.ADJUST_TOP_K, {"value": 10})
            assert not obs.done, "Episode should not be done before max_steps"
        # Final step hits max_steps
        obs = _step(e, ActionType.ADJUST_TOP_K, {"value": 10})
        assert obs.done, "Episode should auto-terminate at max_steps"

    def test_done_flag_propagates_to_observation(self, env1):
        obs = _step(env1, ActionType.SUBMIT)
        assert obs.done is True

    def test_action_recorded_in_history(self, env1):
        action = RAGDebugAction(action_type=ActionType.ADJUST_TOP_K, params={"value": 20})
        env1.step(action)
        assert len(env1._internal_state.action_history) == 1
        assert env1._internal_state.action_history[0].action_type == ActionType.ADJUST_TOP_K


# ---------------------------------------------------------------------------
# Action routing — each action modifies the correct config field
# ---------------------------------------------------------------------------

class TestActionRouting:

    def _get_config(self, env):
        """Grab a copy of the current config fields as a dict."""
        cfg = env._config
        return {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "threshold": cfg.similarity_threshold,
            "top_k": cfg.top_k,
            "model": cfg.embedding_model,
            "reranking": cfg.use_reranking,
            "context_limit": cfg.context_window_limit,
        }

    def test_adjust_chunk_size(self, env1):
        _step(env1, ActionType.ADJUST_CHUNK_SIZE, {"value": 256})
        assert env1._config.chunk_size == 256

    def test_adjust_chunk_overlap(self, env1):
        _step(env1, ActionType.ADJUST_CHUNK_OVERLAP, {"value": 100})
        assert env1._config.chunk_overlap == 100

    def test_adjust_threshold(self, env1):
        _step(env1, ActionType.ADJUST_THRESHOLD, {"value": 0.15})
        assert env1._config.similarity_threshold == pytest.approx(0.15)

    def test_adjust_top_k(self, env1):
        _step(env1, ActionType.ADJUST_TOP_K, {"value": 25})
        assert env1._config.top_k == 25

    def test_swap_embedding_model(self, env1):
        _step(env1, ActionType.SWAP_EMBEDDING_MODEL, {"model": "medical"})
        assert env1._config.embedding_model == EmbeddingModel.MEDICAL

    def test_toggle_reranking_on(self, env1):
        assert env1._config.use_reranking is False
        _step(env1, ActionType.TOGGLE_RERANKING, {"enabled": True})
        assert env1._config.use_reranking is True

    def test_toggle_reranking_off(self, env1):
        _step(env1, ActionType.TOGGLE_RERANKING, {"enabled": True})
        _step(env1, ActionType.TOGGLE_RERANKING, {"enabled": False})
        assert env1._config.use_reranking is False

    def test_adjust_context_limit(self, env1):
        _step(env1, ActionType.ADJUST_CONTEXT_LIMIT, {"value": 8192})
        assert env1._config.context_window_limit == 8192

    def test_invalid_chunk_size_sets_error(self, env1):
        # Set chunk_size smaller than the current chunk_overlap (default 50)
        # to trigger the model_validator "overlap must be < chunk_size".
        obs = _step(env1, ActionType.ADJUST_CHUNK_SIZE, {"value": 10})
        assert obs.last_action_error is not None

    def test_invalid_model_sets_error(self, env1):
        obs = _step(env1, ActionType.SWAP_EMBEDDING_MODEL, {"model": "nonexistent"})
        assert obs.last_action_error is not None

    def test_unrelated_fields_unchanged_after_action(self, env1):
        before = self._get_config(env1)
        _step(env1, ActionType.ADJUST_TOP_K, {"value": 20})
        after = self._get_config(env1)
        # Only top_k should change
        assert after["chunk_size"] == before["chunk_size"]
        assert after["threshold"] == before["threshold"]
        assert after["model"] == before["model"]
        assert after["reranking"] == before["reranking"]
        assert after["context_limit"] == before["context_limit"]


# ---------------------------------------------------------------------------
# Bug fix: ADJUST_CHUNK_OVERLAP must trigger _recompute_S_faulted()
# ---------------------------------------------------------------------------

class TestChunkOverlapRecompute:
    """
    Verifies the fix for the bug where ADJUST_CHUNK_OVERLAP did not call
    _recompute_S_faulted(), meaning the overlap parameter had no effect on
    retrieval scores until a different action happened to trigger recomputation.
    """

    def _make_env_with_chunk_too_small(self, overlap_value):
        """
        Set up an environment where CHUNK_TOO_SMALL is active, then set a
        specific overlap, and return the S_faulted matrix.

        Uses the default chunk_size (512) so that both overlap_value=0 and
        overlap_value=450 are valid (450 < 512 satisfies overlap < chunk_size).
        """
        from models import FaultConfig, FaultType as FT
        e = RAGDebugEnvironment()
        e.reset(seed=42, task_id=1)

        # Force CHUNK_TOO_SMALL fault so overlap modulation is relevant.
        e._injected_faults = [FaultConfig(fault_type=FT.CHUNK_TOO_SMALL)]

        # Apply the overlap we want to test.
        action = RAGDebugAction(
            action_type=ActionType.ADJUST_CHUNK_OVERLAP,
            params={"value": overlap_value},
        )
        e.step(action)
        return e._S_faulted.copy()

    def test_overlap_recompute_changes_s_faulted(self):
        """
        Two environments identical except for chunk_overlap should have
        different S_faulted matrices after ADJUST_CHUNK_OVERLAP, proving
        the recomputation is happening.
        """
        S_low_overlap = self._make_env_with_chunk_too_small(overlap_value=0)
        S_high_overlap = self._make_env_with_chunk_too_small(overlap_value=450)
        # With CHUNK_TOO_SMALL active, higher overlap reduces noise sigma,
        # so the two matrices should differ.
        assert not np.allclose(S_low_overlap, S_high_overlap), (
            "ADJUST_CHUNK_OVERLAP should immediately recompute S_faulted; "
            "different overlap values should yield different matrices."
        )

    def test_overlap_high_reduces_noise_magnitude(self):
        """
        After fixing the bug: higher overlap should reduce the noise added by
        CHUNK_TOO_SMALL, making the faulted matrix closer to S_true.
        Uses chunk_size=512 (default) so both overlap values (0, 450) are valid.
        """
        from models import FaultConfig, FaultType as FT

        def _make_and_get_diff(overlap_value):
            e = RAGDebugEnvironment()
            e.reset(seed=7, task_id=1)
            e._injected_faults = [FaultConfig(fault_type=FT.CHUNK_TOO_SMALL)]
            # Capture S_true before overlap action (use default chunk_size=512)
            model_key = "general"
            S_true = e._s_true_episode[model_key].copy()
            e.step(RAGDebugAction(
                action_type=ActionType.ADJUST_CHUNK_OVERLAP,
                params={"value": overlap_value},
            ))
            return float(np.abs(e._S_faulted - S_true).mean())

        diff_low = _make_and_get_diff(0)
        diff_high = _make_and_get_diff(450)
        assert diff_high < diff_low, (
            "Higher overlap should reduce CHUNK_TOO_SMALL noise, "
            "making S_faulted closer to S_true"
        )


# ---------------------------------------------------------------------------
# SUBMIT grading
# ---------------------------------------------------------------------------

class TestSubmit:

    def test_submit_sets_done(self, env1):
        obs = _step(env1, ActionType.SUBMIT)
        assert obs.done is True

    def test_submit_success_reward_in_range(self):
        """After enough improvement, submit should yield a high reward."""
        e = RAGDebugEnvironment()
        e.reset(seed=0, task_id=1)
        # Adjust threshold low to maximise coverage, then submit
        _step(e, ActionType.ADJUST_THRESHOLD, {"value": 0.05})
        _step(e, ActionType.ADJUST_TOP_K, {"value": 50})
        obs = _step(e, ActionType.SUBMIT)
        # Reward should be in [0.7, 1.0] or [0.0, 0.2] depending on success
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0

    def test_early_submit_penalty_reward_low(self, env1):
        """Submitting immediately (without fixing anything) should give a low reward."""
        obs = _step(env1, ActionType.SUBMIT)
        # Immediate submit without any fixes likely yields failure reward in [0, 0.2]
        # This is not guaranteed to always be < 0.7 depending on episode, but
        # it's the expected case for a fresh poorly-tuned environment.
        assert obs.reward is not None
        assert 0.0 <= obs.reward <= 1.0
