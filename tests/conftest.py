import sys
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.auth.credentials import AnonymousCredentials

# ---------------------------------------------------------------------------
# Global: prevent ADC / credential errors during import of app modules
# ---------------------------------------------------------------------------
mock_default = unittest.mock.MagicMock(return_value=(AnonymousCredentials(), "dummy-project-id"))
unittest.mock.patch("google.auth.default", mock_default).start()


# ---------------------------------------------------------------------------
# Session-scoped autouse: intercept live Gemini API calls in all tests.
#
# generate_advice() in app/agent.py calls:
#   client.aio.models.generate_content(model=..., contents=...)
# We patch the underlying _api_client's async request to return a canned
# GenerateContentResponse so no real network call is made.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True, scope="session")
def mock_genai_generate_content():
    """Stub out all async Gemini generate_content calls with a fake response."""
    # Build a minimal fake response that genai will accept
    fake_response = MagicMock()
    fake_response.text = (
        "Wheat is commonly affected by aphids and fungal rust. "
        "Use crop rotation and avoid excess nitrogen fertiliser. "
        "Organic options include neem oil and insecticidal soap."
    )

    async def _fake_generate(*args, **kwargs):
        return fake_response

    # Patch at the genai SDK layer so it covers both Gemini API Studio and
    # Vertex AI paths used in generate_advice().
    with patch(
        "google.genai.models.AsyncModels.generate_content",
        new=_fake_generate,
    ):
        yield
