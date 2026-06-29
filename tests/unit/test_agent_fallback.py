# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest.mock
from unittest.mock import MagicMock, patch
import pytest
from app.agent import generate_advice

@pytest.mark.asyncio
async def test_generate_advice_fallback() -> None:
    """Test that generate_advice falls back to alternative models if the primary model fails."""
    # Set up a mock context
    ctx = MagicMock()
    ctx.state = {
        "original_query": "What are best practices for growing wheat?",
        "crop": "wheat",
        "advice": ""
    }
    ctx.route = None

    fake_response = MagicMock()
    fake_response.text = "Here is the fallback advice for wheat."

    call_count = 0
    called_models = []

    async def mock_generate_content(model, contents):
        nonlocal call_count
        call_count += 1
        called_models.append(model)
        if call_count == 1:
            raise RuntimeError("503 Service Unavailable")
        return fake_response

    from app.agent import model
    # Patch the generate_content method on the API client
    with patch.object(model.api_client.aio.models, "generate_content", new=mock_generate_content):
        result = await generate_advice(ctx, None)
        assert result == "Here is the fallback advice for wheat."
        assert call_count == 2
        assert called_models == ["gemini-2.5-flash", "gemini-2.0-flash"]
        assert ctx.state["advice"] == "Here is the fallback advice for wheat."


def test_scrub_sensitive_data() -> None:
    """Test that scrub_sensitive_data correctly redacts phone numbers and Aadhaar IDs."""
    from app.agent import scrub_sensitive_data
    
    # Test case 1: Aadhaar formatted and unformatted, and standard 10-digit/formatted phone numbers
    ctx = MagicMock()
    ctx.state = {
        "original_query": "My phone is 9876543210 and my Aadhaar ID is 1234 5678 9012. Please advise on rice crop."
    }
    
    result = scrub_sensitive_data(ctx, None)
    assert "[REDACTED_PHONE]" in result
    assert "[REDACTED_AADHAAR]" in result
    assert "9876543210" not in result
    assert "1234 5678 9012" not in result
    assert ctx.state["original_query"] == result

    # Test case 2: No sensitive data
    ctx2 = MagicMock()
    ctx2.state = {
        "original_query": "Please provide advice on wheat crop pest control."
    }
    result2 = scrub_sensitive_data(ctx2, None)
    assert result2 == "Please provide advice on wheat crop pest control."
    assert ctx2.state["original_query"] == "Please provide advice on wheat crop pest control."
