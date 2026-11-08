# Enterprise AI Customer Support Agent

An example customer-support agent built with LangGraph and LangChain. It retrieves policy information from a local knowledge base, looks up order status through an optional REST API, and pauses refund requests for human approval.

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key"
python support_agent.py
```

Set `OPENAI_MODEL` to use a different model (defaults to `gpt-4o-mini`). The interactive application keeps conversation context for the current process. Type `quit` to exit.

## Knowledge base

Edit [`knowledge_base.json`](knowledge_base.json) to provide your own policy and FAQ documents. The included entries are demonstration policies and must be replaced with approved company information before use. Retrieval uses a lightweight keyword-overlap baseline and should be replaced with an evaluated vector or search index for production workloads.

## Order lookup integration

Configure `ORDER_API_URL` with the base URL of an order service that accepts `GET /{order_id}` and returns JSON containing a string `status`; optional `tracking_number` and `estimated_delivery` strings are also displayed. HTTPS is required except for localhost development. If the service uses bearer-token authentication, set `ORDER_API_TOKEN`. Requests time out after five seconds, and failures are surfaced as a controlled support response. The lookup function can also be replaced with an application-specific integration via `create_support_graph(order_lookup=...)`.

## Human review

Refund requests suspend the LangGraph run and emit a `refund_approval` interrupt. The CLI asks a reviewer to approve or reject it; no refund is issued automatically. Applications embedding the graph can inspect `__interrupt__` and resume with `Command(resume={"approved": True})` or `Command(resume={"approved": False})`. Keep a persistent LangGraph checkpointer and stable per-conversation `thread_id` when deploying beyond this in-memory demo.

## Tests

Run the offline workflow and retrieval tests with:

```bash
python -m unittest discover -s tests -v
```

The tests do not require API credentials or an order service. Production use should add a real order-system adapter and evaluate response grounding, retrieval quality, escalation decisions, tool failures, and latency with reproducible cases.
