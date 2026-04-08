# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Rag Debug Env Environment.

This module creates an HTTP server that exposes the RagDebugEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from models import RAGDebugAction, RAGDebugObservation
from server.rag_debug_env_environment import RagDebugEnvironment


# Create the app with web interface and README integration
app = create_app(
    RagDebugEnvironment,
    RAGDebugAction,
    RAGDebugObservation,
    env_name="rag_debug_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str | None = None, port: int | None = None):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m server.app

    Args:
        host: Optional host address. If omitted, parsed from CLI args.
        port: Optional port number. If omitted, parsed from CLI args.

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn server.app:app --workers 4
    """
    if host is None or port is None:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int, default=7860)
        args = parser.parse_args()
        host = args.host if host is None else host
        port = args.port if port is None else port

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
