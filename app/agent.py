# ruff: noqa
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

import datetime
from zoneinfo import ZoneInfo
import os
import re
from typing import Any

from dotenv import load_dotenv
import google.auth
from google.adk.workflow import Workflow, START, FunctionNode, Edge
from google.adk.events import RequestInput
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.agents.context import Context
from google.genai import types
from pydantic import BaseModel, Field

# Load local environment variables (.env)
load_dotenv()

# Setup Google Cloud defaults if available, otherwise fallback to Gemini API Studio
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    pass

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

# If GEMINI_API_KEY is present, prioritize Gemini API Studio, otherwise use Vertex AI
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Initialize the Gemini model configuration
model = Gemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
)

# 1. State schema for maintaining workflow state across pauses and executions
class FarmerAdvisoryState(BaseModel):
    original_query: str = Field(default="")
    crop: str = Field(default="")
    advice: str = Field(default="")
    approved_advice: str = Field(default="")

# 2. Graph Node Function Implementations
def init_query(ctx: Context, node_input: Any) -> str:
    """Extracts and stores the initial query in session state."""
    query_text = ""
    if isinstance(node_input, str):
        query_text = node_input
    elif isinstance(node_input, types.Content):
        query_text = "".join(part.text for part in node_input.parts if part.text)
    else:
        query_text = str(node_input)
    ctx.state["original_query"] = query_text
    return query_text

def triage_crop(ctx: Context, node_input: str) -> Any:
    """Checks if a crop is defined. Requests crop from the user if missing."""
    # Check if we are resuming from a request_crop interrupt
    if "request_crop" in ctx.resume_inputs:
        crop_input = ctx.resume_inputs["request_crop"]
        ctx.state["crop"] = crop_input
        ctx.route = "has_crop"
        return f"Crop set to: {crop_input}"

    crop = ctx.state.get("crop")
    if not crop:
        # Attempt to parse crop from the original query
        query = ctx.state.get("original_query", "").lower()
        for c in ["wheat", "rice", "corn", "maize", "tomato", "potato", "soybean"]:
            if c in query:
                ctx.state["crop"] = c
                ctx.route = "has_crop"
                return f"Identified crop: {c}"

        # Crop is missing. Trigger HITL by yielding RequestInput
        ctx.route = "need_crop"
        return RequestInput(
            interrupt_id="request_crop",
            message="What crop are you farming? Please specify (e.g. wheat, rice, corn).",
        )

    ctx.route = "has_crop"
    return f"Crop already set to: {crop}"

def scrub_sensitive_data(ctx: Context, node_input: Any) -> str:
    """Scrubs phone numbers and Aadhaar-style 12-digit numbers from the original query."""
    query = ctx.state.get("original_query", "")

    # Aadhaar-style: 12-digit number (typically formatted as 4-4-4 or 12 digits)
    aadhaar_pattern = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{12}\b')

    # Phone numbers: matches common phone formats (e.g. +91XXXXXXXXXX, 10-digit number \b\d{10}\b, etc.)
    phone_pattern = re.compile(r'\+?\d{1,3}[-\s\.]?\(?\d{3}\)?[-\s\.]?\d{3}[-\s\.]?\d{4}\b|\b\d{10}\b')

    redacted = False
    cleaned_query = query

    if aadhaar_pattern.search(cleaned_query):
        cleaned_query = aadhaar_pattern.sub("[REDACTED_AADHAAR]", cleaned_query)
        redacted = True

    if phone_pattern.search(cleaned_query):
        cleaned_query = phone_pattern.sub("[REDACTED_PHONE]", cleaned_query)
        redacted = True

    if redacted:
        import logging
        logging.warning("Security Checkpoint: Sensitive personal data (phone/Aadhaar) detected and redacted.")
        ctx.state["original_query"] = cleaned_query

    return cleaned_query

async def generate_advice(ctx: Context, node_input: Any) -> str:
    """Uses the Gemini model to generate a custom agricultural advisory report."""
    crop = ctx.state.get("crop")
    query = ctx.state.get("original_query", "general farming practices")

    client = model.api_client
    prompt = (
        f"You are an expert agricultural advisor. The farmer is asking about '{crop}' "
        f"with the following query: '{query}'. Provide detailed agricultural advice, "
        f"crop management tips, and best practices. If relevant, mention natural/organic "
        f"treatments. If chemical pesticides are required, explicitly list them."
    )

    models_to_try = [model.model, "gemini-2.0-flash", "gemini-3.5-flash"]
    response = None
    last_error = None

    for m in models_to_try:
        try:
            response = await client.aio.models.generate_content(
                model=m,
                contents=prompt,
            )
            break
        except Exception as e:
            last_error = e
            import logging
            logging.warning(f"Failed to generate advice using model variant {m}: {e}. Trying fallback...")
            continue

    if response is None:
        raise last_error

    advice = response.text
    ctx.state["advice"] = advice

    # Check if advice recommends chemical pesticides, routing to review if so
    if "pesticide" in advice.lower() or "chemical" in advice.lower():
        ctx.route = "requires_review"
    else:
        ctx.route = "direct_output"

    return advice

def expert_review(ctx: Context, node_input: Any) -> Any:
    """Expert review interrupt node for pesticide recommendations."""
    if "expert_review" in ctx.resume_inputs:
        review_response = ctx.resume_inputs["expert_review"]
        ctx.state["approved_advice"] = review_response
        return review_response

    advice = ctx.state.get("advice", "")
    message = (
        f"WARNING: Pesticide recommendations detected for crop: {ctx.state.get('crop')}. "
        f"Please review and modify/approve this advice:\n\n{advice}"
    )
    return RequestInput(
        interrupt_id="expert_review",
        message=message,
    )

def finalize_advice(ctx: Context, node_input: Any) -> str:
    """Prepares and returns the final advice report."""
    approved = ctx.state.get("approved_advice")
    if approved:
        return approved
    return ctx.state.get("advice", "No advice available.")

# 3. Configure FunctionNode definitions (rerun_on_resume=True for HITL nodes)
init_node = FunctionNode(func=init_query, name="init_node")
triage_node = FunctionNode(func=triage_crop, name="triage_node", rerun_on_resume=True)
security_node = FunctionNode(func=scrub_sensitive_data, name="security_node")
advisor_node = FunctionNode(func=generate_advice, name="advisor_node")
expert_review_node = FunctionNode(func=expert_review, name="expert_review_node", rerun_on_resume=True)
finalize_node = FunctionNode(func=finalize_advice, name="finalize_node")

# 4. Construct workflow edges
edges = [
    (START, init_node),
    (init_node, triage_node),
    Edge(from_node=triage_node, to_node=security_node, route="has_crop"),
    (security_node, advisor_node),
    Edge(from_node=advisor_node, to_node=expert_review_node, route="requires_review"),
    Edge(from_node=advisor_node, to_node=finalize_node, route="direct_output"),
    (expert_review_node, finalize_node),
]

# Compile Workflow as root_agent
root_agent = Workflow(
    name="root_agent",
    edges=edges,
    state_schema=FarmerAdvisoryState,
)

# Initialize application containing the workflow
app = App(
    root_agent=root_agent,
    name="app",
)
