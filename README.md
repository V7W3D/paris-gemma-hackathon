# Claim Verifier

A claim-verification chat app built on two cooperating agents. You paste a statement, and the
system decomposes it into checkable claims, researches them against a document corpus served over
the Alien MCP server, judges each claim against what it found, and answers with a verdict and
citations.

The point of the project is the context engineering: a second agent owns everything the reasoning
agent sees. It assembles the context object and persists a new revision after every decision, and
that is its whole job.

## How it works

```mermaid
flowchart TD
  UI["React chat + history"] -->|/api| API["FastAPI"]
  API --> Workflow["ClaimVerificationWorkflow"]
  Workflow -->|"1. assemble"| Ctx["ContextAgent (agent 2)"]
  Workflow -->|"2. decide"| Ver["VerifierAgent (agent 1)"]
  Ctx -->|"context object"| Ver
  Ver -->|"3. queries"| Workflow
  Workflow -->|"4. search"| Alien["Alien MCP server"]
  Workflow -->|"5. commit"| Ctx
  Ctx --> Mongo[("MongoDB")]
  Ver --> Brev["Gemma on NVIDIA Brev"]
  Ctx --> Brev
```

### The five decision points

| Stage | Agent 1 decides | Retrieval |
| --- | --- | --- |
| `decompose` | the atomic claims in the message | none |
| `plan` | what evidence would settle each claim | none |
| `gather` | which searches to run against the corpus | one search per query |
| `assess` | supported / refuted / insufficient, per claim | none |
| `verdict` | the final answer, label and confidence | none |

The gather stage loops until agent 1 says it has enough or the search budget runs out.

### The two agents

- **`VerifierAgent`** ([backend/services/agent/verifier_agent.py](backend/services/agent/verifier_agent.py))
  sees exactly two things: the memory file
  ([backend/memory/verifier_memory.md](backend/memory/verifier_memory.md)) and the context object.
  It never talks to the database and never picks a tool — at `gather` it writes queries, and the
  orchestrator runs them.
- **`ContextAgent`** ([backend/services/agent/context_agent.py](backend/services/agent/context_agent.py))
  never answers the user and never judges a claim. It reads the conversation, `assemble`s the
  context for a turn, `commit`s each decision plus the passages its searches returned into a new
  revision, and `compact`s the turn into a running summary for the next one.

The first call of a conversation gets an empty context object; from there every call carries what
agent 2 decided to keep.

### Retrieval over MCP

MCP is the transport to one search tool, not a surface the model chooses from.
[`AlienRetriever`](backend/services/retrieval/alien_client.py) connects to the Alien MCP server at
startup, lists the tools it advertises, and picks the one to search with — the highest-ranked
candidate by name, or whatever `ALIEN_MCP_SEARCH_TOOL` pins. Arguments are filled from that tool's
own JSON schema (the query, and a result limit and dataset ids when it accepts them), and whatever
shape it answers with is normalised into passages. Nothing is hardcoded to one deployment.

### Why prompted JSON instead of function calling

Gemma has no dependable native tool-calling, so nothing relies on the OpenAI `tools` field. Each
stage has a Pydantic output model in
[backend/models/schemas/actions.py](backend/models/schemas/actions.py) that Pydantic AI injects
into the prompt as a JSON schema and validates on the way back, re-asking the model when it
misbehaves. Requests carry a single system message, which is what the Gemma chat template accepts.

## Running it

You need Python 3.10+ and Node 20+. Everything else — MongoDB, the Brev endpoint, the Alien MCP
server — is optional; see [Nothing configured?](#nothing-configured-it-still-runs) below.

Run both commands from the repository root, in two shells:

```bash
# 1. backend  -> http://localhost:8000
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cp backend/.env.example backend/.env      # then fill in the credentials you have
.venv/bin/python -m uvicorn backend.main:app --reload --port 8000

# 2. frontend -> http://localhost:5173, proxies /api to :8000
npm install --prefix frontend
npm run dev --prefix frontend
```

Open http://localhost:5173. `GET /api/status` reports what is live and what is mocked.

### Where everything goes

| What | Where it goes |
| --- | --- |
| Python dependencies | declared in `backend/requirements.txt`, installed into `.venv/` at the repository root |
| Node dependencies | declared in `frontend/package.json`, installed into `frontend/node_modules/` |
| Credentials and settings | `backend/.env`, copied from [backend/.env.example](backend/.env.example) — a `.env` at the repository root works too |
| Agent 1's durable knowledge | `backend/memory/verifier_memory.md`, edited by hand |

The backend is imported as the `backend` package, so `uvicorn` has to be started from the
repository root, not from inside `backend/`.

### Nothing configured? It still runs

Each dependency degrades on its own, so the full workflow is demonstrable offline with an empty
`.env`:

| Missing | Behaviour |
| --- | --- |
| `BREV_BASE_URL` | a deterministic mock model answers every stage |
| `ALIEN_MCP_URL` | searches return placeholder passages |
| MongoDB | conversations and contexts live in memory for the process lifetime |

## Configuration

All settings live in `backend/.env` (see [backend/.env.example](backend/.env.example)).

| Variable | Purpose |
| --- | --- |
| `BREV_BASE_URL` | OpenAI-compatible base URL of the Gemma deployment, including `/v1` |
| `BREV_API_KEY` | bearer token for that endpoint |
| `BREV_MODEL` | model name to request, e.g. `google/gemma-3-27b-it` |
| `MONGODB_URI`, `MONGODB_DB` | conversation and context storage |
| `ALIEN_MCP_URL` | streamable HTTP endpoint of the Alien MCP server |
| `ALIEN_MCP_TOKEN` | bearer token for that server |
| `ALIEN_MCP_SEARCH_TOOL` | pin the tool to search with; empty means auto-detect |
| `ALIEN_DATASET_IDS` | datasets to scope searches to, when the tool accepts them |
| `ALIEN_SEARCH_LIMIT` | passages requested per search |
| `MOCK_LLM`, `MOCK_SEARCH` | force mocking even when credentials exist |
| `MAX_GATHER_STEPS` | search budget per turn |
| `MAX_CLAIMS` | claims extracted per message |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/status` | what is wired up and what is mocked |
| `GET` | `/api/chats` | conversation history for the sidebar |
| `POST` | `/api/chats` | start a conversation |
| `GET` | `/api/chats/{id}` | full conversation with messages, verdicts and traces |
| `PATCH` | `/api/chats/{id}` | rename |
| `DELETE` | `/api/chats/{id}` | delete the conversation and its contexts |
| `GET` | `/api/chats/{id}/contexts` | every context revision agent 2 wrote |
| `POST` | `/api/chats/{id}/messages` | run a turn; SSE by default, `?stream=false` for JSON |

The SSE stream emits `turn_started`, `stage`, `claims`, `retrieval`, `token`, `message`, `error`
and `done` events, which is what drives the live reasoning trail in the UI.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite covers the output-envelope parsing and salvage, agent 2's context commits, tool
discovery and result normalisation against an in-process MCP server, a full workflow run over the
mock model, and the HTTP API including the SSE stream. It never touches the network.

## Layout

```
backend/
  main.py, config.py, dependencies.py   app, settings, wiring
  db/                                   Mongo store + in-memory fallback
  memory/verifier_memory.md             durable knowledge for agent 1
  models/schemas/                       chat, context object, stage outputs
  routers/                              chats CRUD, verification stream
  services/agent/                       the two agents, prompts, inference
  services/retrieval/                   the Alien MCP client
  services/workflow/                    the five decision points
frontend/src/
  features/chat, features/history, features/trace
  hooks/, lib/                          SSE client and types
```
