# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Centralized Vertex AI configuration for VLM and VLA bridges.

Single source of truth for project ID, location, and client initialization.
Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION in .env or environment
to authenticate once for all Vertex AI services.
"""

import os

_CACHED_CLIENT = None
_CACHED_PROJECT = None
_CACHED_LOCATION = None


def get_vertex_project() -> str | None:
    """Project ID from env. None if not configured."""
    return os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_GENAI_PROJECT")


def get_vertex_location() -> str:
    """Vertex AI region. Default us-central1."""
    return os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_GENAI_LOCATION") or "us-central1"


def get_vertex_client():
    """Lazy singleton Vertex AI client (google-genai).

    Returns:
        genai.Client configured for Vertex AI, or None if project/libraries missing.
    """
    global _CACHED_CLIENT, _CACHED_PROJECT, _CACHED_LOCATION

    project = get_vertex_project()
    location = get_vertex_location()
    if not project:
        return None

    if (
        _CACHED_CLIENT is not None
        and _CACHED_PROJECT == project
        and _CACHED_LOCATION == location
    ):
        return _CACHED_CLIENT

    try:
        from google import genai

        _CACHED_CLIENT = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        _CACHED_PROJECT = project
        _CACHED_LOCATION = location
        return _CACHED_CLIENT
    except ImportError:
        return None
