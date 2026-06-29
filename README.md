# Farmer Advisory Agent 🌾

An ADK 2.0 multi-agent system that gives smallholder farmers in India quick, practical crop advisory support — with built-in privacy protection and human-in-the-loop safety review for sensitive recommendations.

Built for Kaggle's **AI Agents: Intensive Vibe Coding Capstone Project** (Agents for Good track).

## The Problem

Smallholder farmers, especially in regions like Andhra Pradesh, often lack timely access to agricultural advice. A farmer noticing yellow spots on tomato leaves may have to wait days for an extension officer visit, rely on guesswork, or get advice from a non-specialist — while the problem spreads.

This project explores how an AI agent can provide an immediate first layer of triage and advice, while staying honest about its limits: it escalates uncertain or sensitive cases (like pesticide recommendations) to a human reviewer rather than acting fully autonomously, and protects any personal information a farmer might share.

**Scope note:** This submission is built and demonstrated for Indian smallholder farmers (Telugu/English context). The underlying architecture — intent routing, security screening, human-in-the-loop escalation — is designed to be extensible to other regions and languages, but that extensibility is intentionally future work, not something this version claims to have validated.

## Why Agents

A single LLM prompt can't safely handle this use case end-to-end:
- It needs to **route** different kinds of queries differently (a clear crop question vs. an off-topic one)
- It needs to **screen inputs** before they reach the model (personal data protection)
- It needs to **pause for human review** when the advice involves something sensitive (chemical pesticides)

A graph-based multi-agent workflow makes each of these a distinct, inspectable, testable step — rather than hoping one big prompt handles every edge case correctly every time.

## Architecture

```
START
  │
  ▼
init_node          (parses incoming farmer query)
  │
  ▼
triage_node        (identifies the crop mentioned, if any)
  │
  ├── no crop recognized ──────────────► RequestInput (human-in-the-loop)
  │
  └── has_crop
       │
       ▼
  security_node     (redacts phone numbers / Aadhaar-style IDs
       │             before any text reaches the LLM)
       ▼
  advisor_node       (Gemini LLM generates crop advisory)
       │
       ├── pesticide/chemical advice detected ──► expert_review_node
       │                                          (RequestInput — pause for
       │                                           human approval before
       │                                           finalizing)
       │
       └── no sensitive content ─────────────────► finalize_node
                                                          │
                                                          ▼
                                                         END
```

### Key components

- **`triage_node`** — Plain-code keyword detection for known crops. No LLM call needed for this step — keeps routing fast and cheap, reserving the LLM for genuine advisory reasoning.
- **`security_node`** — Regex-based redaction of phone numbers and Aadhaar-style 12-digit IDs from the query *before* it reaches the LLM or any logs. Logs that redaction occurred, never the sensitive value itself.
- **`advisor_node`** — Gemini-powered agent that generates structured crop advice (diagnosis possibilities, organic and chemical treatment options, preventive practices).
- **`expert_review_node`** — A `RequestInput` human-in-the-loop checkpoint. Any advice mentioning pesticides/fungicides is paused here for human approval before being shown to the farmer. In a real deployment, this step would route to agricultural extension officers or Krishi Vigyan Kendra (KVK) staff — an existing network of government agricultural support centers in India.

## Course Concepts Demonstrated

| Concept | Where |
|---|---|
| Multi-agent system (ADK 2.0 graph Workflow) | `app/agent.py` — nodes, edges, conditional routing |
| Security features | `security_node` — PII redaction before LLM exposure |
| Human-in-the-loop / Deployability | `RequestInput` interrupts for unclear queries and pesticide-advice review |

## Project Structure

```
farmer-advisory-agent/
├── app/
│   ├── agent.py               # Graph workflow: triage, security, advisor, review nodes
│   ├── agent_runtime_app.py   # Runtime/serving entry point
│   └── app_utils/
├── tests/
│   ├── unit/                  # Node-level tests (incl. PII redaction tests)
│   ├── integration/           # End-to-end workflow tests
│   └── eval/                  # LLM-as-judge evaluation dataset/config
├── pyproject.toml
└── README.md
```

## Setup & Running Locally

**Requirements:**
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- A [Google AI Studio API key](https://aistudio.google.com/app/apikey)

**1. Clone and install:**
```bash
git clone https://github.com/siri-k08/farmer-advisory-agent.git
cd farmer-advisory-agent
uv sync
```

**2. Set up your API key:**

Create a `.env` file in the project root (this file is gitignored and never committed):
```
GEMINI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT=dummy-project-id
```

**3. Run the tests:**
```bash
uv run pytest tests/unit tests/integration
```

**4. Try it interactively:**
```bash
uv run python -m google.adk.cli web app
```
Open the printed local URL (typically `http://127.0.0.1:8080/dev-ui/?app=app`), select `app` from the dropdown, and try:
- *"My tomato plants have yellow spots on the leaves, what should I do?"* — watch the security and advisor nodes run, and the pesticide-review pause trigger.
- *"My phone number is 9876543210, my tomato plants have yellow spots, what should I do?"* — watch the phone number get redacted before the advisor responds.

## Honest Limitations

- **Human reviewer is simulated in this demo.** In this submission, escalations are reviewed by the developer as a stand-in. A production deployment would route these to real agricultural extension staff (e.g., KVK officers).
- **Regional scope.** Advisory content and crop-keyword detection are tuned for common Indian crops; this is not validated for other regions' agriculture.
- **Not a replacement for professional diagnosis.** The agent explicitly recommends consulting a local agricultural extension office for confirmation on serious or ambiguous cases.

## License

This project is submitted under the Kaggle Capstone competition rules. See competition rules for licensing terms applicable to winning submissions (CC-BY 4.0).
