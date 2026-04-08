"""
tests/test_stdout_format.py
---------------------------
Tests for the mandatory stdout logging format defined in inference.py.

The evaluation harness parses [START], [STEP], and [END] lines from stdout.
Any deviation in field names, ordering, or formatting causes incorrect scoring.

These tests parse the actual output of the logging functions and verify:
- Field names match exactly.
- Fields appear in the required order.
- Numeric formatting is correct (reward: 2dp, score: 3dp).
- Boolean values are lowercase.
- No newlines appear within a log line.
"""

import re
import sys
import io
import pytest

# Patch sys path so inference.py can be imported from the project root.
import importlib.util
from pathlib import Path

# Import the logging functions directly from inference.py
_INFERENCE_PATH = Path(__file__).parent.parent / "inference.py"
_spec = importlib.util.spec_from_file_location("inference", _INFERENCE_PATH)
_inf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inf)

log_start = _inf.log_start
log_step = _inf.log_step
log_end = _inf.log_end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_stdout(fn, *args, **kwargs) -> str:
    """Capture what fn() prints to stdout and return it as a string."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue().rstrip("\n")


def _parse_fields(line: str) -> dict:
    """
    Parse a log line like '[TAG] key=val key2=val2 ...' into a dict.
    Values can contain letters, digits, underscores, dots, commas, hyphens,
    slashes, colons, and curly braces — everything up to the next space=word pair.
    """
    # Strip tag
    line = re.sub(r"^\[.*?\]\s*", "", line)
    # Tokenise: split on whitespace-separated key=value pairs
    pattern = re.compile(r"(\w+)=(\S+)")
    return {m.group(1): m.group(2) for m in pattern.finditer(line)}


# ---------------------------------------------------------------------------
# [START] line
# ---------------------------------------------------------------------------

class TestStartLine:

    def test_prefix(self):
        line = _capture_stdout(log_start, task="task_1", env="rag_debug_env",
                                model="Qwen/Qwen2.5-72B")
        assert line.startswith("[START]"), f"Expected [START] prefix, got: {line!r}"

    def test_field_names(self):
        line = _capture_stdout(log_start, task="task_1", env="rag_debug_env",
                                model="Qwen/Qwen2.5-72B")
        fields = _parse_fields(line)
        assert "task" in fields, "Missing 'task' field"
        assert "env" in fields, "Missing 'env' field"
        assert "model" in fields, "Missing 'model' field"

    def test_field_order(self):
        line = _capture_stdout(log_start, task="task_1", env="rag_debug_env",
                                model="Qwen/Qwen2.5-72B")
        # Verify positional ordering via index search
        assert line.index("task=") < line.index("env=") < line.index("model="), (
            "Fields must appear in order: task, env, model"
        )

    def test_field_values(self):
        line = _capture_stdout(log_start, task="task_1", env="rag_debug_env",
                                model="Qwen/Qwen2.5-72B")
        fields = _parse_fields(line)
        assert fields["task"] == "task_1"
        assert fields["env"] == "rag_debug_env"

    def test_single_line(self):
        line = _capture_stdout(log_start, task="task_1", env="rag_debug_env",
                                model="Qwen/Qwen2.5-72B")
        assert "\n" not in line, "Log line must not contain internal newlines"


# ---------------------------------------------------------------------------
# [STEP] line
# ---------------------------------------------------------------------------

class TestStepLine:

    def _step_line(self, step=1, action="submit()", reward=0.0, done=False, error=None):
        return _capture_stdout(log_step, step=step, action=action,
                               reward=reward, done=done, error=error)

    def test_prefix(self):
        assert self._step_line().startswith("[STEP]")

    def test_field_names(self):
        line = self._step_line()
        fields = _parse_fields(line)
        for field in ("step", "action", "reward", "done", "error"):
            assert field in fields, f"Missing '{field}' field in [STEP] line"

    def test_field_order(self):
        line = self._step_line()
        positions = {
            "step": line.index("step="),
            "action": line.index("action="),
            "reward": line.index("reward="),
            "done": line.index("done="),
            "error": line.index("error="),
        }
        ordered = sorted(positions, key=positions.get)
        assert ordered == ["step", "action", "reward", "done", "error"], (
            f"Field order wrong: {ordered}"
        )

    def test_reward_two_decimal_places(self):
        line = self._step_line(reward=0.123456)
        m = re.search(r"reward=(\d+\.\d+)", line)
        assert m is not None, "reward field not found"
        assert len(m.group(1).split(".")[1]) == 2, (
            f"reward should have 2 decimal places, got: {m.group(1)!r}"
        )

    def test_reward_exact_format(self):
        line = self._step_line(reward=0.5)
        assert "reward=0.50" in line, f"Expected reward=0.50, got: {line!r}"

    def test_done_false_lowercase(self):
        line = self._step_line(done=False)
        assert "done=false" in line, "done=False must be serialized as 'false'"

    def test_done_true_lowercase(self):
        line = self._step_line(done=True)
        assert "done=true" in line, "done=True must be serialized as 'true'"

    def test_error_null_when_none(self):
        line = self._step_line(error=None)
        assert "error=null" in line, "null error must be serialized as 'null'"

    def test_error_string_when_present(self):
        line = self._step_line(error="Invalid value")
        assert "error=" in line
        assert "null" not in line.split("error=")[1].split()[0], (
            "error field should contain the message, not 'null'"
        )

    def test_single_line(self):
        line = self._step_line(action="adjust_threshold(value=0.15)")
        assert "\n" not in line

    @pytest.mark.parametrize("reward", [0.0, 0.5, 1.0, 0.123])
    def test_various_rewards_in_range(self, reward):
        line = self._step_line(reward=reward)
        m = re.search(r"reward=(\d+\.\d{2})", line)
        assert m is not None, f"reward field missing or wrong format for reward={reward}"
        val = float(m.group(1))
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# [END] line
# ---------------------------------------------------------------------------

class TestEndLine:

    def _end_line(self, success=True, steps=5, score=0.85, rewards=None):
        rewards = rewards or [0.3, 0.5, 0.7, 0.85, 0.90]
        return _capture_stdout(log_end, success=success, steps=steps,
                               score=score, rewards=rewards)

    def test_prefix(self):
        assert self._end_line().startswith("[END]")

    def test_field_names(self):
        line = self._end_line()
        fields = _parse_fields(line)
        for field in ("success", "steps", "score", "rewards"):
            assert field in fields, f"Missing '{field}' field in [END] line"

    def test_field_order(self):
        line = self._end_line()
        positions = {
            "success": line.index("success="),
            "steps": line.index("steps="),
            "score": line.index("score="),
            "rewards": line.index("rewards="),
        }
        ordered = sorted(positions, key=positions.get)
        assert ordered == ["success", "steps", "score", "rewards"], (
            f"Field order wrong: {ordered}"
        )

    def test_success_true_lowercase(self):
        line = self._end_line(success=True)
        assert "success=true" in line

    def test_success_false_lowercase(self):
        line = self._end_line(success=False)
        assert "success=false" in line

    def test_score_three_decimal_places(self):
        line = self._end_line(score=0.85)
        m = re.search(r"score=(\d+\.\d+)", line)
        assert m is not None, "score field not found"
        assert len(m.group(1).split(".")[1]) == 3, (
            f"score should have 3 decimal places, got: {m.group(1)!r}"
        )

    def test_rewards_comma_separated(self):
        line = self._end_line(rewards=[0.3, 0.5, 0.7])
        m = re.search(r"rewards=(\S+)", line)
        assert m is not None, "rewards field not found"
        parts = m.group(1).split(",")
        assert len(parts) == 3

    def test_rewards_two_decimal_places(self):
        line = self._end_line(rewards=[0.123, 0.456])
        m = re.search(r"rewards=(\S+)", line)
        assert m is not None
        for part in m.group(1).split(","):
            decimal_part = part.split(".")[1] if "." in part else ""
            assert len(decimal_part) == 2, (
                f"Each reward in rewards should have 2 decimal places, got: {part!r}"
            )

    def test_score_in_unit_interval(self):
        for score in [0.0, 0.5, 1.0]:
            line = self._end_line(score=score)
            m = re.search(r"score=(\d+\.\d+)", line)
            assert m is not None
            assert 0.0 <= float(m.group(1)) <= 1.0

    def test_single_line(self):
        line = self._end_line()
        assert "\n" not in line

    def test_empty_rewards_list(self):
        """Edge case: no steps taken should produce empty rewards."""
        line = self._end_line(rewards=[])
        assert "rewards=" in line
