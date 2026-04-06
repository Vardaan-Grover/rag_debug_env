# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rag Debug Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import RAGDebugAction, RAGDebugObservation


class RAGDebugEnv(
    EnvClient[RAGDebugAction, RAGDebugObservation, State]
):
    """
    Client for the Rag Debug Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with RagDebugEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_description)
        ...
        ...     result = client.step(RAGDebugAction(action_type="adjust_chunk_size", params={"value": 1024}))
        ...     print(result.observation.metrics)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = RagDebugEnv.from_docker_image("rag_debug_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(RAGDebugAction(action_type="adjust_chunk_size", params={"value": 1024}))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RAGDebugAction) -> Dict:
        """
        Convert RAGDebugAction to JSON payload for step message.

        Args:
            action: RAGDebugAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(mode='json')

    def _parse_result(self, payload: Dict) -> StepResult[RAGDebugObservation]:
        """
        Parse server response into StepResult[RAGDebugObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with RAGDebugObservation
        """
        obs_data = payload.get("observation", {})
        observation = RAGDebugObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
