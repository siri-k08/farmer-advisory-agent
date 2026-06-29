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

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent


def test_agent_stream() -> None:
    """
    Integration test for the ADK 2.0 Workflow agent stream functionality.

    When the query contains no recognized crop keyword, triage_node issues a
    RequestInput HITL interrupt — the runner emits an adk_request_input
    function-call event rather than plain text.  We assert at least one event
    arrives (the interrupt) and that it carries either a function_call or text
    part (both are valid content shapes for workflow events).
    """
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # Query without a crop keyword → triggers triage_node HITL interrupt
    message = types.Content(
        role="user", parts=[types.Part.from_text(text="Why is the sky blue?")]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one event from the workflow"

    # The triage_node pauses with an adk_request_input function-call event.
    has_function_call_or_text = any(
        event.content
        and event.content.parts
        and any(p.function_call is not None or p.text is not None for p in event.content.parts)
        for event in events
    )
    assert has_function_call_or_text, (
        "Expected a function_call (RequestInput) or text event from the workflow"
    )


def test_agent_stream_with_crop_in_query() -> None:
    """
    When the query names a crop, triage_node resolves it without a HITL
    interrupt and the workflow moves on to advice generation.
    """
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user2", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="I have a wheat crop. What pests should I watch out for?")],
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="test_user2",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected at least one event"

    # ADK 2.0 FunctionNode return values are emitted on event.output (a raw
    # Python value), not on event.content.  Accept either field as evidence
    # that the workflow produced a text advisory.
    has_text_content = False
    for event in events:
        # Check content.parts for text (e.g. LLM streaming chunks)
        if (
            event.content
            and event.content.parts
            and any(part.text for part in event.content.parts)
        ):
            has_text_content = True
            break
        # Check output for plain string returned by FunctionNode (e.g. finalize_node)
        if isinstance(event.output, str) and event.output.strip():
            has_text_content = True
            break
    assert has_text_content, (
        "Expected at least one event with text content (in content.parts or event.output)"
    )
