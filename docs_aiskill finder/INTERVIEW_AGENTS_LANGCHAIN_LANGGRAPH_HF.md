# Interview Guide: LangChain, LangGraph, Hugging Face, Tools & Agents

**Project:** RByte.ai / AI Skill Finder  
**Purpose:** One document so you can understand these topics from zero and explain exactly what this project does (and does not do) in interviews.

Read this top to bottom once. Then use **Section 8 (Interview Scripts)** before calls.

---

## 0. Honest project truth (read this first)

| Topic | In this project? | Where | Interview-safe claim |
|---|---|---|---|
| **LangGraph** | Yes — real | `backend/app/services/recommendation/langgraph_agent.py` | “We use a LangGraph state machine for the learning-path agent.” |
| **LangChain** | Yes — light | Same file (`ChatOpenAI`, messages) + notebook curator | “We use LangChain as the LLM wrapper inside the LangGraph agent, not as full LCEL chains everywhere.” |
| **Agents** | Yes — graph agent | Recommendation learning path | “We built a multi-step recommendation agent with shared state and nodes.” |
| **Tools / function calling** | Partially — *manual tools as graph nodes*, not OpenAI `tools=` API | `search_courses` node = DB tool | “Course search is a deterministic tool-like step in the graph. We did **not** wire OpenAI native function calling yet.” |
| **Hugging Face** | As **market skill data only**, not as a runtime library | `role_requirements.json` emerging skills | “HF Transformers is a skill we benchmark candidates against. Runtime embeddings use OpenAI, not Hugging Face models.” |

**Do not claim in interviews:**
- “We use Hugging Face Transformers in production inference.”
- “We use LangChain Agents / AgentExecutor with tool calling.”
- “Every LLM call goes through LangChain.”

Most GPT pipelines (resume NER, benchmark, questions, rubric, gaps, resume writer) use the **OpenAI Python SDK directly**, not LangChain.

---

## 1. Big picture: how GenAI fits in RByte.ai

```
User journey (what the product does)
────────────────────────────────────
Resume upload
   → GPT-4o extracts skills          (OpenAI SDK)
   → Benchmark vs role requirements  (OpenAI SDK + DB)
   → Adaptive assessment             (OpenAI SDK)
   → Rubric scoring                  (OpenAI SDK)
   → Gap analysis                    (OpenAI SDK)
   → Learning path agent             (LangGraph + LangChain ChatOpenAI)
   → ATS resume writer               (OpenAI SDK)

Where the 4 topics sit
──────────────────────
LangChain  → thin LLM interface inside the agent
LangGraph  → orchestrates the multi-step learning path
Tools      → “search courses in Postgres” (graph node, not tool_calls API)
Agents     → the recommendation agent itself
Hugging Face → skill name in role data (ML/AI roles), not runtime stack
```

### End-to-end architecture (interview diagram)

```
┌──────────────┐     REST/JWT      ┌─────────────────────────────┐
│  Next.js UI  │ ───────────────►  │  FastAPI backend            │
│ Learning Path│                   │  POST /recommendation/generate
└──────────────┘                   └──────────────┬──────────────┘
                                                  │
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ compute_gaps()               │
                                   │ (benchmark + assessment)     │
                                   └──────────────┬───────────────┘
                                                  │ gaps[]
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ LangGraph Recommendation     │
                                   │ Agent (StateGraph)           │
                                   │                              │
                                   │  analyse → search → rank     │
                                   │         → validate → END     │
                                   │                              │
                                   │  rank node uses LangChain    │
                                   │  ChatOpenAI (GPT-4o)         │
                                   └──────────────┬───────────────┘
                                                  │ learning_path JSON
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ LearningPath table (Postgres)│
                                   └──────────────────────────────┘
```

---

## 2. Topic A — LangChain (beginner → project)

### 2.1 What is LangChain? (simple)

**LangChain** is a Python/JS framework that makes it easier to build apps around LLMs.

Think of it as Lego blocks for LLM apps:
- **Models** — talk to GPT / Claude / local models with one API style
- **Prompts** — templates for system/user messages
- **Output parsers** — turn model text into JSON / Pydantic objects
- **Chains** — pipe steps: prompt → model → parse
- **Tools / agents** — let the model call functions (search, DB, APIs)
- **Memory / retrievers** — chat history, RAG, etc.

Without LangChain you can still call OpenAI’s SDK. LangChain is an **orchestration helper**, not a model.

```
Without LangChain:     Your code ──► openai.chat.completions.create()
With LangChain:        Your code ──► ChatOpenAI.invoke([messages])
                                     (same GPT underneath)
```

### 2.2 LangChain vs OpenAI SDK (interview clarity)

| | OpenAI SDK | LangChain |
|---|---|---|
| What it is | Official client for OpenAI APIs | Framework that can wrap many model providers |
| In our project | Used in **most** services | Used mainly in **recommendation agent** (+ curator notebook) |
| Best for | Direct, simple GPT calls | Agents, multi-step graphs, swapping providers later |

**Why we still use both:**  
Simple one-shot pipelines (extract JSON, score answer) are cleaner with OpenAI SDK. Multi-step agent orchestration is cleaner with LangGraph, which sits on LangChain message/model types.

### 2.3 What we actually use from LangChain

**Production file:** `backend/app/services/recommendation/langgraph_agent.py`

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    model=settings.openai_model,   # gpt-4o
    temperature=0,
    api_key=settings.openai_api_key,
)

response = llm.invoke([
    SystemMessage(content="You are a learning path architect..."),
    HumanMessage(content=f"Target role: {state['target_role']} ..."),
])
```

**Offline notebook:** `backend/role_requirements_curator.ipynb`
- `ChatOpenAI` + `ChatPromptTemplate` + `JsonOutputParser`
- Used to curate role skill requirements (data prep), not live API traffic.

**Dependencies** (`requirements.txt`):
- `langchain==0.2.1`
- `langchain-openai==0.1.8`
- `langgraph==0.1.1`
- `langsmith==0.1.77` (listed; tracing not fully wired as product feature yet)

### 2.4 Why we added LangChain

1. **Required glue for LangGraph** — LangGraph nodes naturally use LangChain chat models and message types.
2. **Consistent message format** — `SystemMessage` / `HumanMessage` keep prompts clear.
3. **Future flexibility** — same agent code can swap providers more easily than raw SDK everywhere.
4. **Interview / product narrative** — agentic learning path is a differentiator vs “just GPT prompts.”

### 2.5 What is left / what could replace LangChain

| Current | Could replace with | When it makes sense |
|---|---|---|
| `ChatOpenAI` in agent | Raw `openai` client | If you want fewer dependencies |
| Notebook LCEL-ish prompt/parser | Same as production OpenAI + Pydantic | Keep stack consistent |
| Full LangChain AgentExecutor | Not used today — don’t claim it | Only if you add true tool-calling agents |

**Honest line for interview:**  
“We use LangChain lightly as the chat model layer inside LangGraph, not as a heavy LCEL app across the whole backend.”

---

## 3. Topic B — LangGraph (beginner → project)

### 3.1 What is LangGraph? (simple)

**LangGraph** is a library (from the LangChain team) for building **stateful, multi-step agents** as a **graph / state machine**.

- **Node** = one step of work (function)
- **Edge** = what runs next
- **State** = shared dict that every node reads/writes
- **Conditional edge** = if/else routing (retry, escalate, stop)

```
Simple chain (LangChain):     A → B → C   (always same path)

LangGraph:                    A → B → C
                                    ↘ (if fail) back to B
                                    ↘ (if ok) END
```

**Why graphs matter:** Learning paths need decisions — “enough courses?” “over 80 hours?” “retry search?” — not only one prompt.

### 3.2 Core concepts you must know

| Concept | Meaning | In our code |
|---|---|---|
| `StateGraph` | Builder for the agent graph | `StateGraph(RecommendationState)` |
| `TypedDict` state | Shared memory | `RecommendationState` |
| `add_node` | Register a step | `analyse_gaps`, `search_courses`, `rank_courses`, `validate_path` |
| `add_edge` | Fixed next step | analyse → search → rank → validate |
| `add_conditional_edges` | Branching | `should_retry` → search again or END |
| `compile()` | Make runnable agent | `graph.compile()` |
| `ainvoke(state)` | Run async | `await agent.ainvoke(initial_state)` |

### 3.3 Our recommendation agent (exact design)

**File:** `backend/app/services/recommendation/langgraph_agent.py`  
**API:** `POST /recommendation/generate` in `backend/app/api/routes/recommendation.py`

#### State (`RecommendationState`)

```
gaps, target_role, seniority, experience_years
searched_skills, found_courses, search_attempts
learning_path, total_hours, estimated_weeks, hours_per_week
reasoning, error
```

#### Graph flow (as implemented in code)

```
                    ┌─────────────────┐
                    │  analyse_gaps   │  Filter missing/below_bar
                    │  (top 6 skills) │  Sort by priority_score
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ search_courses  │  Postgres Course table
                    │  (DB "tool")    │  match skill_axes, limit 2 each
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  rank_courses   │  If no courses → friendly message
                    │  (GPT via LC)   │  Else GPT-4o orders ≤6 courses,
                    │                 │  ≤80h, 10h/week rules
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ validate_path   │  Hard caps: max 6 courses, 80h,
                    │                 │  weeks = hours/10 (cap 12)
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ should_retry(state)         │
              │ if error==insufficient_courses│
              │    and attempts < 2         │
              │    → search_courses again   │
              │ else → END                  │
              └─────────────────────────────┘
```

#### Node responsibilities (say this in interviews)

1. **`analyse_gaps`** — deterministic Python; no LLM. Picks top 6 missing/below-bar skills.
2. **`search_courses`** — deterministic DB query; acts like a **tool**.
3. **`rank_courses`** — LLM brain; uses LangChain `ChatOpenAI` to sequence courses and explain why. Has JSON parse fallback if GPT returns bad JSON.
4. **`validate_path`** — business rules / guardrails; no LLM.

### 3.4 Why LangGraph instead of a simple GPT call?

| Need | Simple one-shot GPT | LangGraph |
|---|---|---|
| Multi-step workflow | Hard to structure | Natural nodes |
| Mix code + LLM | Awkward | Easy (DB search + GPT rank) |
| Guardrails (80h, 6 courses) | Easy to forget | Dedicated validate node |
| Retry / branching | Manual if/else spaghetti | Conditional edges |
| Debuggability | One giant prompt | Per-node logs (`Agent: ...`) |

**Interview answer (short):**  
“Learning path needs conditional multi-step logic: analyse gaps, search catalogue, LLM-rank, validate timeline. LangGraph gives us an explicit state machine. A single prompt cannot reliably own DB search + business caps + retries.”

### 3.5 Gap between docs and code (be honest if asked)

`project.md` sometimes says nodes like `build_learning_path` / `estimate_timeline` and “retry with broader terms / GPT-suggested resources.”

**Actual code today:**
- Nodes are: `analyse_gaps` → `search_courses` → `rank_courses` → `validate_path`
- Retry condition exists (`should_retry`) but **`error` is almost never set to `"insufficient_courses"`** in current nodes, so the retry path is **scaffolded more than actively used**
- Empty catalogue → clear user message; it does **not** invent GPT courses outside the catalogue (by design — anti-hallucination)

That honesty shows maturity: “We designed for retry; current empty-catalogue path fails closed with a message instead of hallucinating courses.”

### 3.6 What is left / replace options for LangGraph

| Idea | Status | Why |
|---|---|---|
| Broader skill synonym search on retry | Left / incomplete | `should_retry` ready; search not broadened yet |
| GPT-suggested external resources fallback | Documented, not coded | Would risk hallucination; catalogue-only is safer |
| LangSmith tracing of each node | Dependency present; product wiring left | Great for prompt debugging |
| Replace LangGraph with plain functions | Possible | Lose clear agent narrative + conditional structure |
| Replace with CrewAI / AutoGen / Semantic Kernel | Possible later | Overkill for 4-node workflow today |

---

## 4. Topic C — Tools & Function Calling (beginner → project)

### 4.1 What is function calling / tools? (simple)

LLMs only generate text. **Tools** let them take actions:

```
User: "Find courses for Docker and build a 4-week plan."

Without tools:
  Model guesses course names from memory → hallucinations.

With tools:
  Model: I should call search_courses(skill="Docker")
  App runs real DB/API → returns real courses
  Model: Now I'll order these real courses into weeks
```

**OpenAI native function calling** looks like:
1. You register tools (JSON schemas: name, parameters)
2. Model returns `tool_calls` instead of final answer
3. Your code executes the function
4. You send results back; model finishes

**LangChain tools** wrap the same idea with `@tool` / `StructuredTool` and agent loops.

### 4.2 Two ways to give an agent “tools”

```
Pattern 1 — Native tool calling (LLM decides when to call)
─────────────────────────────────────────────────────────
LLM ──tool_calls──► search_courses(skill)
     ◄── results─── your code
LLM ──final answer─► learning path

Pattern 2 — Graph nodes as tools (developer decides when)
─────────────────────────────────────────────────────────
Node search_courses ALWAYS runs (SQL)
Node rank_courses ALWAYS runs (LLM sees results)
```

### 4.3 What THIS project does

We use **Pattern 2** (graph nodes as tools), **not** Pattern 1.

| Step | Is it a “tool”? | Who decides to run it? |
|---|---|---|
| `search_courses` (SQL on `Course`) | Yes — tool-like | Graph always runs it after analyse |
| `rank_courses` (GPT) | Reasoning step | Graph always runs it after search |
| OpenAI `tools=` / `tool_calls` | **No** | Not implemented |
| LangChain `@tool` / `ToolNode` | **No** | Not implemented |

So in interviews say:

> “Our agent’s tool is course-catalogue search implemented as a LangGraph node. The LLM does not choose tools via function calling; the graph enforces the workflow so search always happens before ranking. That is more reliable for a product path with hard business rules.”

### 4.4 Why this design (and when native tools would be better)

**Why graph-enforced tools are good here**
- Guarantees DB search before GPT
- Prevents GPT from inventing courses
- Easy to enforce 80h / 6-course caps in Python
- Cheaper / more predictable than multi-round tool loops

**When you might add real function calling later**
- Agent must choose among many tools (web search, calendar, email, RAG, payments)
- Dynamic number of tool rounds
- User chat interface (“ask anything about my gaps”)

### 4.5 Diagram: tools in our agent vs textbook tool-calling agent

```
Textbook ReAct / function-calling agent
───────────────────────────────────────
         ┌────────────┐
    ┌───►│    LLM     │──┐
    │    └────────────┘  │
    │         │          │ tool_calls?
    │         ▼          ▼
    │    final answer   execute tool
    │         │          │
    └─────────┴──────────┘  (loop)

Our recommendation agent
────────────────────────
 analyse (code) → search (DB tool) → rank (LLM) → validate (code) → END
        fixed pipeline; LLM only ranks known courses
```

### 4.6 What is left for tools / function calling

| Enhancement | Why useful |
|---|---|
| Convert `search_courses` to LangChain `@tool` + `bind_tools` | Closer to industry “tool calling” demos |
| Add `web_search` / Udemy API tool | Fill catalogue gaps without hallucinating |
| Add `get_user_profile` tool | Chat-style career coach agent |
| Structured outputs / JSON schema mode in rank node | Replace manual JSON cleanup |

---

## 5. Topic D — Agents (beginner → project)

### 5.1 What is an AI agent? (simple)

An **agent** is a system where an LLM (plus code/tools) **plans and acts over multiple steps** toward a goal, using **state**, not just one prompt → one answer.

```
Chatbot (not really an agent):
  Q → LLM → A

Agent:
  Goal → think → act (tool) → observe → think → act → ... → finish
```

Common agent patterns:
- **ReAct** — Reason + Act loop with tools
- **Plan-and-execute** — plan first, then run steps
- **State machine / graph agent** — LangGraph style (what we use)
- **Multi-agent** — researcher + writer + critic (CrewAI etc.) — **not in this project**

### 5.2 Is our recommendation system an agent?

**Yes — with nuance.**

It is an **agentic workflow / graph agent**:
- Has a goal: build a personalised learning path
- Has memory/state: `RecommendationState`
- Has actions: search catalogue, call LLM, validate
- Has (designed) branching: conditional retry

It is **not** a free-form chat agent that invents its own plan every turn.

**Best interview phrasing:**  
“We implemented a LangGraph multi-step agent for learning-path generation — a controlled state machine rather than an open-ended ReAct loop.”

### 5.3 How the agent is triggered in the product

```
Frontend Learning Path page
        │
        ▼
POST /recommendation/generate
  { profile_id, benchmark_id, session_id? }
        │
        ▼
If path already saved → return it (idempotent)
Else:
  compute_gaps(benchmark, session)
  run_recommendation_agent(gaps, role, seniority, db)
  save LearningPath row
  return path_steps + reasoning + hours/weeks
```

### 5.4 Agent quality / guardrails you should mention

1. **Catalogue-grounded** — GPT may only order courses from `found_courses`
2. **Working-professional realism** — max ~10h/week, path under 80 hours, max 6 courses
3. **Weak vs strong learner** — prompt branches on assessment score (&lt;5 vs &gt;7)
4. **Deterministic caps in `validate_path`** — even if GPT over-recommends, Python trims
5. **JSON fallback** — if parse fails, priority-ordered catalogue list still returned
6. **Temperature 0** — more stable sequencing

### 5.5 Other “agent-like” pieces that are NOT agents

| Pipeline | Multi-step? | Agent? | Why |
|---|---|---|---|
| Resume NER | No | No | Single structured GPT call |
| Benchmark | No | No | Single scoring call |
| Question generator | No | No | One-shot generation |
| Rubric scorer | No | No | LLM-as-judge, not agent |
| Gap engine | Mostly one GPT analysis | No | Heuristic + one LLM call |
| Recommendation | Yes | **Yes** | LangGraph state machine |

This distinction impresses interviewers.

---

## 6. Topic E — Hugging Face (beginner → project)

### 6.1 What is Hugging Face? (simple)

**Hugging Face** is the main open ecosystem for ML/NLP models:

| Piece | What it is |
|---|---|
| **Hub** | Model + dataset hosting (like GitHub for models) |
| **`transformers`** | Library to run BERT, Llama, Whisper, etc. |
| **`datasets` / `accelerate`** | Data + training helpers |
| **Inference API / Endpoints** | Hosted model APIs |
| **sentence-transformers** | Popular for open-source embeddings |

People say “Hugging Face” to mean: open-source models instead of (or beside) OpenAI.

### 6.2 How HF usually appears in GenAI apps

```
Option A — OpenAI only (our runtime today)
  text → OpenAI embeddings / GPT-4o → product

Option B — Hugging Face models
  text → sentence-transformers / BERT → vectors
  or local Llama via transformers / vLLM

Option C — Hybrid
  GPT-4o for hard reasoning
  HF model for cheap embeddings / classification
```

### 6.3 What THIS project does with Hugging Face

**Runtime code:** does **not** import `transformers`, `huggingface_hub`, or call HF Inference API.

**Data layer:** “Hugging Face Transformers” appears as an **emerging skill** for AI/ML-style roles in:

- `backend/app/data/role_requirements.json`
- mirrored under `app/data/role_requirements.json`

Meaning: when we benchmark an ML / GenAI candidate, the market profile may say HF Transformers is an emerging skill they should know.

**Embeddings in code:** `EmbeddingEngine` uses **OpenAI `text-embedding-3-small`** (1536-d), not HF sentence-transformers.

### 6.4 How to talk about HF without lying

**Good:**  
“In our role requirements for ML/AI engineers, Hugging Face Transformers is curated as an emerging skill. Our production LLM and embeddings stack is OpenAI GPT-4o + text-embedding-3-small. Hugging Face would be a natural next step if we want open-source embeddings or on-prem models to cut cost.”

**Bad:**  
“We built the platform on Hugging Face.”

### 6.5 If interviewer asks “why not Hugging Face embeddings?”

Possible strong answers:
1. **Speed to ship** — OpenAI embeddings + pgvector already work with Azure/OpenAI keys we have
2. **Ops** — hosting HF models needs GPU/CPU capacity, batching, versioning
3. **Quality for skill phrases** — OpenAI embeddings are strong out of the box for short skill strings
4. **Roadmap** — HF `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) is a clear cost-optimization swap for the embedding engine interface we already abstracted (`EmbeddingEngine`)

### 6.6 Where HF *could* replace something later

| Current | Possible HF replacement | Benefit |
|---|---|---|
| `text-embedding-3-small` | `sentence-transformers` / HF Inference | Lower cost, offline option |
| GPT-4o for simple classification | Small HF classifier | Cheaper triage |
| Nothing for local demo | Tiny Llama via transformers | Offline demos |
| Skill name only in JSON | Keep as skill; optional HF course content | Product content, not infra |

---

## 7. How the four topics connect (one mental model)

```
                    ┌─────────────────────────────────────┐
                    │              AGENTS                 │
                    │  “multi-step goal-seeking systems”  │
                    └─────────────────┬───────────────────┘
                                      │
                         implemented with
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │             LANGGRAPH               │
                    │   state machine: nodes + edges      │
                    │   RecommendationState + compile()   │
                    └───────────────┬─────────────────────┘
                                    │
              uses chat model from  │  tool-like node =
                                    ▼
         ┌──────────────┐    ┌────────────────────┐
         │  LANGCHAIN   │    │ TOOLS / ACTIONS    │
         │ ChatOpenAI   │    │ search_courses SQL │
         │ Messages     │    │ (not tool_calls)   │
         └──────┬───────┘    └────────────────────┘
                │
                ▼
         GPT-4o (OpenAI)

HUGGING FACE in this project
────────────────────────────
Not in this runtime loop.
Appears as a *skill to assess* in role requirements,
and as a future alternative for embeddings/models.
```

### Dependencies relationship

```
langgraph  ──depends on──►  langchain-core / langchain messaging
langchain-openai ──wraps──►  OpenAI Chat Completions
langsmith (optional) ──────►  tracing/observability

huggingface ──not a pip dependency in requirements.txt──
```

---

## 8. Interview scripts (memorize these)

### 8.1 30-second GenAI pitch (agents angle)

> “I worked on RByte.ai, a GenAI skills platform. We use GPT-4o across resume extraction, benchmarking, assessment scoring, and gap analysis. For personalised learning paths we built a **LangGraph agent**: it analyses skill gaps, searches our course catalogue like a tool, asks GPT-4o to sequence courses for working professionals, then validates hours and weeks with hard business rules. LangChain is the chat-model layer inside that agent; most other pipelines use the OpenAI SDK directly for simplicity.”

### 8.2 “Explain LangChain vs LangGraph”

> “LangChain is the toolkit — models, prompts, parsers, tools. LangGraph is specifically for **stateful multi-step agents** as graphs. In our app, LangGraph owns the learning-path workflow; LangChain’s `ChatOpenAI` is used inside the ranking node.”

### 8.3 “Did you use agents / tools?”

> “Yes. The recommendation service is a graph agent with shared state. Course search is a deterministic tool-like node hitting Postgres. We intentionally did **not** use open-ended function-calling loops, because learning paths need guaranteed order: search first, then LLM rank, then validate caps. That’s more reliable for production.”

### 8.4 “Why not only one GPT prompt for learning path?”

> “One prompt would mix retrieval and planning and often invent courses. We separate concerns: Python finds real courses; GPT only orders and explains; Python enforces max 6 courses and 80 hours. LangGraph makes that separation explicit.”

### 8.5 “Do you use Hugging Face?”

> “As market skill data for ML/AI roles — Hugging Face Transformers is listed as an emerging skill in our curated requirements. Our runtime models are OpenAI. The embedding engine is abstracted so we could swap in sentence-transformers later for cost.”

### 8.6 “What would you improve next?”

Pick 2–3:
1. Wire **LangSmith** tracing on each LangGraph node  
2. Implement real **retry with skill synonyms** when catalogue misses  
3. Optional **native tool calling** for a career-coach chat agent  
4. Optional **HF embeddings** behind `EmbeddingEngine`  
5. Broader course catalogue so the agent has more to recommend  

### 8.7 Trap questions — safe answers

| Trap | Safe answer |
|---|---|
| “Show me your LangChain chain” | “We don’t use a long LCEL chain app-wide; LC is ChatOpenAI inside LangGraph.” |
| “Which HF model fine-tuned?” | “We didn’t fine-tune HF models; HF appears in skill taxonomy.” |
| “How does AgentExecutor work in your repo?” | “We don’t use AgentExecutor; we use LangGraph StateGraph.” |
| “Where is function calling?” | “Graph-node tools today; OpenAI tools API is a roadmap item.” |

---

## 9. File map (study these before interview)

| File | Why it matters |
|---|---|
| `backend/app/services/recommendation/langgraph_agent.py` | **Main evidence** — LangGraph + LangChain + agent |
| `backend/app/api/routes/recommendation.py` | How API triggers the agent |
| `backend/app/models/recommendation.py` | `LearningPath`, course model, agent metadata fields |
| `backend/app/data/seed_courses.py` | Catalogue the agent searches |
| `backend/role_requirements_curator.ipynb` | Extra LangChain usage (curation) |
| `backend/app/services/benchmark/embedding_engine.py` | OpenAI embeddings (HF alternative point) |
| `backend/app/data/role_requirements.json` | Where “Hugging Face Transformers” skill lives |
| `backend/requirements.txt` | Package versions for stack discussion |
| `project.md` § Module 7 + Interview guide | Product narrative (cross-check vs code) |

---

## 10. Mini glossary (quick revision)

| Term | One-line meaning |
|---|---|
| LLM | Large language model (e.g. GPT-4o) |
| Prompt | Instructions + context you send the model |
| Chain | Fixed pipeline of prompt → model → parse |
| Agent | Multi-step system that can use tools/state toward a goal |
| Tool | Function/API the agent can invoke (search, DB, web) |
| Function calling | Model returns structured “call this function” instead of only text |
| State | Shared memory across agent steps |
| LangGraph | Graph/state-machine framework for agents |
| LangChain | Broader LLM app framework |
| LangSmith | Tracing/eval platform for LangChain/LangGraph |
| Hugging Face | Open model hub + `transformers` ecosystem |
| Embeddings | Numeric vectors for semantic similarity |
| RAG | Retrieve documents, then generate with that context |
| Hallucination | Model invents facts not in tools/data |
| Temperature 0 | More deterministic outputs |

---

## 11. One-page cheat sheet

```
USED FOR REAL
✔ LangGraph StateGraph recommendation agent
✔ LangChain ChatOpenAI + messages inside rank node
✔ Agent = learning path generator
✔ Tool-like DB course search as a graph node
✔ HF as skill in role requirements JSON

NOT USED (yet / not claim)
✘ Hugging Face runtime models
✘ OpenAI tools= function calling
✘ LangChain AgentExecutor / @tool agents
✘ LangChain for every GPT pipeline
✘ Fully active broader-search retry (scaffold only)

WHY THIS STACK
• Graph = reliable multi-step learning path
• LLM only ranks real courses (anti-hallucination)
• Validate node enforces career-realistic hours
• OpenAI SDK elsewhere = simpler one-shot pipelines

SAY IN INTERVIEW
“LangGraph agent + LangChain chat model + catalogue tool node;
OpenAI SDK for other GPT pipelines; HF as market skill, not infra.”
```

---

## 12. Suggested 20-minute study plan

1. **5 min** — Draw the LangGraph diagram from memory (4 nodes + conditional end).  
2. **5 min** — Open `langgraph_agent.py` and walk each node once.  
3. **5 min** — Rehearse scripts 8.1–8.5 out loud.  
4. **5 min** — Rehearse honesty lines: HF not runtime, no native function calling, LangChain is light.

---

*Generated from the actual RByte.ai / AI Skill Finder codebase. Prefer this document over marketing wording when something conflicts with `langgraph_agent.py`.*
