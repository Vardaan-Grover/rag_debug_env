"""
tests/test_reward.py
--------------------
Tests for the reward computation in RAGDebugEnvironment.

Verifies that:
- All step rewards are bounded in [0.0, 1.0].
- Terminal success rewards are in [0.7, 1.0].
- Terminal failure rewards are in [0.0, 0.2].
- Named components are present and correctly summed.
- Specific reward signals fire in the right direction.
"""

import pytest
import numpy as np

from models import (
    QualityMetrics,
    RAGDebugAction,
    ActionType,
    FaultType,
    FaultConfig,
)
from server.rag_debug_env_environment import RAGDebugEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(coverage=0.5, precision=0.5, empty=0, overflow=0, mh_cov=None):
    return QualityMetrics(
        mean_coverage=coverage,
        mean_precision=precision,
        mean_recall=coverage,
        n_empty_retrievals=empty,
        n_context_overflows=overflow,
        multi_hop_coverage=mh_cov,
    )


def _make_env(task_id=1, seed=42) -> RAGDebugEnvironment:
    env = RAGDebugEnvironment()
    env.reset(seed=seed, task_id=task_id)
    return env


def _dummy_action(action_type=ActionType.ADJUST_TOP_K, params=None):
    return RAGDebugAction(action_type=action_type, params=params or {"value": 15})


# ---------------------------------------------------------------------------
# Reward bounds
# ---------------------------------------------------------------------------

class TestRewardBounds:
    """All step rewards must stay in [0.0, 1.0]."""

    def _check_reward(self, env, prev, new, action):
        r = env._compute_reward(prev, new, action)
        assert 0.0 <= r.value <= 1.0, (
            f"Reward {r.value} out of [0, 1] — components: {r.components}"
        )
        return r

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_perfect_state(self, task_id):
        env = _make_env(task_id)
        prev = _make_metrics(0.9, 0.9)
        new = _make_metrics(1.0, 1.0, mh_cov=1.0 if task_id == 3 else None)
        self._check_reward(env, prev, new, _dummy_action())

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_worst_state(self, task_id):
        env = _make_env(task_id)
        prev = _make_metrics(0.5, 0.5)
        new = _make_metrics(0.0, 0.0, empty=5, mh_cov=0.0 if task_id == 3 else None)
        self._check_reward(env, prev, new, _dummy_action())

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_regression_step(self, task_id):
        env = _make_env(task_id)
        prev = _make_metrics(0.8, 0.7)
        new = _make_metrics(0.2, 0.1, empty=3, overflow=2)
        self._check_reward(env, prev, new, _dummy_action())

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_no_change_step(self, task_id):
        env = _make_env(task_id)
        m = _make_metrics(0.6, 0.5)
        self._check_reward(env, m, m, _dummy_action())

    def test_with_penalties(self):
        env = _make_env(1)
        # Force invalid action error
        env._last_action_error = "some error"
        prev = _make_metrics(0.5, 0.5)
        new = _make_metrics(0.5, 0.5)
        # Also simulate redundancy: same action type as previous
        env._prev_action_type = ActionType.ADJUST_TOP_K
        action = _dummy_action(ActionType.ADJUST_TOP_K)
        r = env._compute_reward(prev, new, action)
        assert 0.0 <= r.value <= 1.0
        assert r.components.get("redundancy_penalty") == -0.04
        assert r.components.get("invalid_action_penalty") == -0.05

    def test_first_step_no_prev(self):
        """First step has no prev metrics; delta_bonus should be 0."""
        env = _make_env(1)
        new = _make_metrics(0.5, 0.5)
        r = env._compute_reward(None, new, _dummy_action())
        assert r.components.get("delta_bonus") == 0.0
        assert 0.0 <= r.value <= 1.0

    def test_reward_bounds_across_many_random_states(self):
        """Fuzz test: random metric combinations should never violate bounds."""
        env = _make_env(1)
        rng = np.random.default_rng(0)
        for _ in range(200):
            cov = float(rng.uniform(0, 1))
            prec = float(rng.uniform(0, 1))
            prev = _make_metrics(float(rng.uniform(0, 1)), float(rng.uniform(0, 1)),
                                  empty=int(rng.integers(0, 5)),
                                  overflow=int(rng.integers(0, 3)))
            new = _make_metrics(cov, prec,
                                empty=int(rng.integers(0, 5)),
                                overflow=int(rng.integers(0, 3)))
            r = env._compute_reward(prev, new, _dummy_action())
            assert 0.0 <= r.value <= 1.0, f"OOB reward={r.value}, cov={cov}, prec={prec}"


# ---------------------------------------------------------------------------
# Terminal rewards
# ---------------------------------------------------------------------------

class TestTerminalRewards:

    def _submit_with_score(self, task_score):
        """Compute terminal reward directly using the same formula as SUBMIT handler."""
        success_value = float(np.clip(0.7 + 0.3 * task_score, 0.7, 1.0))
        failure_value = float(np.clip(0.2 * task_score, 0.0, 0.2))
        return success_value, failure_value

    @pytest.mark.parametrize("task_score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_terminal_success_in_range(self, task_score):
        v, _ = self._submit_with_score(task_score)
        assert 0.7 <= v <= 1.0, f"Success reward {v} outside [0.7, 1.0]"

    @pytest.mark.parametrize("task_score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_terminal_failure_in_range(self, task_score):
        _, v = self._submit_with_score(task_score)
        assert 0.0 <= v <= 0.2, f"Failure reward {v} outside [0.0, 0.2]"

    def test_success_reward_monotone_in_task_score(self):
        scores = np.linspace(0.0, 1.0, 20)
        values = [self._submit_with_score(s)[0] for s in scores]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1)), (
            "Success terminal reward should be monotonically non-decreasing in task_score"
        )

    def test_success_reward_max_at_perfect_score(self):
        v, _ = self._submit_with_score(1.0)
        assert v == pytest.approx(1.0), "Perfect task_score should give reward=1.0"

    def test_success_reward_min_at_threshold(self):
        v, _ = self._submit_with_score(0.0)
        assert v == pytest.approx(0.7), "task_score=0 should give success reward=0.7"


# ---------------------------------------------------------------------------
# Component correctness
# ---------------------------------------------------------------------------

class TestRewardComponents:

    def test_progress_reward_range(self):
        """progress_reward should span [0.10, 0.65]."""
        env = _make_env(1)
        for cov in [0.0, 0.3, 0.6, 0.9, 1.0]:
            r = env._compute_reward(None, _make_metrics(cov, cov), _dummy_action())
            pr = r.components["progress_reward"]
            assert 0.10 <= pr <= 0.65, f"progress_reward={pr} out of [0.10, 0.65] at cov={cov}"

    def test_delta_bonus_positive_on_improvement(self):
        """Improving quality should produce positive delta_bonus."""
        env = _make_env(1)
        prev = _make_metrics(0.3, 0.3)
        new = _make_metrics(0.8, 0.8)
        r = env._compute_reward(prev, new, _dummy_action())
        assert r.components["delta_bonus"] > 0, "Improvement should yield positive delta_bonus"

    def test_delta_bonus_negative_on_regression(self):
        """Regressing quality should produce negative delta_bonus."""
        env = _make_env(1)
        prev = _make_metrics(0.8, 0.8)
        new = _make_metrics(0.2, 0.2)
        r = env._compute_reward(prev, new, _dummy_action())
        assert r.components["delta_bonus"] < 0, "Regression should yield negative delta_bonus"

    def test_step_cost_always_present(self):
        env = _make_env(1)
        r = env._compute_reward(None, _make_metrics(), _dummy_action())
        assert r.components["step_cost"] == -0.01

    def test_empty_retrieval_signal_positive_when_fixed(self):
        """Fixing empty retrievals should yield positive empty_retrieval_signal."""
        env = _make_env(1)
        prev = _make_metrics(0.5, 0.5, empty=4)
        new = _make_metrics(0.5, 0.5, empty=0)
        r = env._compute_reward(prev, new, _dummy_action())
        assert r.components["empty_retrieval_signal"] > 0

    def test_empty_retrieval_signal_negative_when_introduced(self):
        env = _make_env(1)
        prev = _make_metrics(0.5, 0.5, empty=0)
        new = _make_metrics(0.5, 0.5, empty=5)
        r = env._compute_reward(prev, new, _dummy_action())
        assert r.components["empty_retrieval_signal"] < 0

    def test_redundancy_penalty_fires_on_repeat_action(self):
        env = _make_env(1)
        env._prev_action_type = ActionType.ADJUST_THRESHOLD
        action = _dummy_action(ActionType.ADJUST_THRESHOLD, {"value": 0.2})
        r = env._compute_reward(_make_metrics(), _make_metrics(), action)
        assert r.components["redundancy_penalty"] == -0.04

    def test_redundancy_penalty_absent_on_different_action(self):
        env = _make_env(1)
        env._prev_action_type = ActionType.ADJUST_THRESHOLD
        action = _dummy_action(ActionType.ADJUST_TOP_K, {"value": 15})
        r = env._compute_reward(_make_metrics(), _make_metrics(), action)
        assert r.components["redundancy_penalty"] == 0.0


# ---------------------------------------------------------------------------
# Task score formula
# ---------------------------------------------------------------------------

class TestTaskScoreFormula:

    def test_task1_formula(self):
        env = _make_env(1)
        env._state.step_count = 5  # halfway through max 10 steps
        m = _make_metrics(coverage=0.8, precision=0.7)
        score = env._compute_task_score(m)
        efficiency = 1.0 - 5 / 10
        expected = 0.60 * 0.8 + 0.25 * 0.7 + 0.15 * efficiency
        assert score == pytest.approx(expected, abs=1e-6)

    def test_task2_formula(self):
        env = _make_env(2)
        env._state.step_count = 3
        m = _make_metrics(coverage=0.6, precision=0.5)
        score = env._compute_task_score(m)
        efficiency = 1.0 - 3 / 10
        expected = 0.60 * 0.6 + 0.25 * 0.5 + 0.15 * efficiency
        assert score == pytest.approx(expected, abs=1e-6)

    def test_task3_formula(self):
        env = _make_env(3)
        env._state.step_count = 4
        m = _make_metrics(coverage=0.7, precision=0.6, mh_cov=0.55)
        score = env._compute_task_score(m)
        expected = 0.55 * 0.7 + 0.25 * 0.6 + 0.20 * 0.55
        assert score == pytest.approx(expected, abs=1e-6)

    def test_task3_no_multihop_coverage_defaults_to_zero(self):
        env = _make_env(3)
        env._state.step_count = 0
        m = _make_metrics(coverage=0.7, precision=0.6, mh_cov=None)
        score = env._compute_task_score(m)
        expected = 0.55 * 0.7 + 0.25 * 0.6 + 0.20 * 0.0
        assert score == pytest.approx(expected, abs=1e-6)

    def test_perfect_task1_score(self):
        env = _make_env(1)
        env._state.step_count = 0
        m = _make_metrics(coverage=1.0, precision=1.0)
        score = env._compute_task_score(m)
        assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Success check
# ---------------------------------------------------------------------------

class TestSuccessCheck:

    def test_task1_passes_at_threshold(self):
        env = _make_env(1)
        env._state.step_count = 0
        m = _make_metrics(1.0, 1.0)
        score = env._compute_task_score(m)
        assert env._check_success(m, score)

    def test_task1_fails_below_threshold(self):
        env = _make_env(1)
        env._state.step_count = 9
        m = _make_metrics(0.3, 0.3)
        score = env._compute_task_score(m)
        assert not env._check_success(m, score)

    def test_task3_requires_multihop_coverage(self):
        env = _make_env(3)
        env._state.step_count = 0
        # High coverage but low multi-hop — should fail
        m = _make_metrics(1.0, 1.0, mh_cov=0.4)
        score = env._compute_task_score(m)
        assert not env._check_success(m, score)

    def test_task3_passes_when_both_conditions_met(self):
        env = _make_env(3)
        env._state.step_count = 0
        m = _make_metrics(1.0, 1.0, mh_cov=0.8)
        score = env._compute_task_score(m)
        assert env._check_success(m, score)
