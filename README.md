# MedRoute

Intelligent Medical Triage and Routing Agent for Low-Resource Clinics

AMD Developer Hackathon ACT II -- Track 3 (Unicorn Track)

Voice-first, multilingual medical triage agent. Patients speak Urdu, Hindi, or English; Whisper Large V3 Turbo transcribes locally with native Urdu support and Roman Urdu normalization; a deterministic LangChain orchestrator reasons over symptoms with WHO-grounded RAG; model inference uses OpenRouter's free-model router for low-cost testing. A formal safety governance layer with 16 hard red-flag overrides runs before any LLM is invoked.

## Architecture

```
Voice Input --> Whisper Large V3 Turbo --> Input Parser (symptoms, clusters, duration)
                                         |
                                  Safety Pre-Check (16 hard red flags)
                                         |  (halt on match -> emergency)
                                  Complexity Scorer
                                  (syndrome clusters + calibrated confidence)
                                         |
                +----------------------------------------------+
                |  Deterministic Route Orchestrator            |
                |  (routing policy is code, not free-form LLM) |
                |                                              |
                |  plan by route:                              |
                |    local_only      -> OpenRouter             |
                |    local_with_rag  -> RAG -> OpenRouter      |
                |    remote          -> RAG -> OpenRouter      |
                |    escalation_bias -> RAG -> OpenRouter      |
                |                                              |
                |  cascade on failure:                         |
                |    provider down -> clinical heuristics ->   |
                |    clinician escalation                      |
                |                                              |
                |  confidence fusion: scorer (union) model     |
                +----------------------------------------------+
                                         |
                          PDF Report + urgency + cascade audit
```

**Design principles** (aligned with hybrid CDSS / CLARITY-style systems):
- LLMs support diagnosis; **routing and safety are deterministic**
- Complexity-aware grounding: simple cases call the model directly; complex cases add retrieved evidence
- Conservative under uncertainty (escalation bias + confidence fusion)
- Works offline: seed RAG + clinical heuristics when GPU/API unavailable

## Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 0 | Voice Input | Whisper Large V3 Turbo transcribes Urdu/Hindi/English audio locally through faster-whisper. Native Urdu script is normalized to Roman Urdu for the clinic UI. |
| 1 | Input Parser | Extracts structured symptoms, symptom clusters, patient age, pregnancy status, and symptom duration from free-text transcript. Maps Urdu/Hindi terms via Roman Urdu transliteration and Devanagari script handling. |
| 2 | Safety Pre-Check | 16 hard red-flag patterns checked against parsed input. Any match halts the pipeline immediately -- no LLM query. Covers ACS, FAST stroke, sepsis, anaphylaxis, PE, meningitis, SAH, DKA, GI bleed, seizure, suicide crisis, head trauma, obstetric emergency, infant fever, sick infant, and severe dehydration. |
| 3 | Complexity Scorer | Scores the case on a 0-10+ scale with syndrome cluster matching, duration awareness, vagueness penalty, and patient context multipliers (age extremes, pregnancy). Outputs a recommended route and calibrated confidence. |
| 4 | Deterministic Orchestrator | Executes a fixed tool plan based on the scored route. Uses OpenRouter directly for simple cases and adds RAG evidence for moderate/complex cases. Falls back to deterministic heuristics and clinician escalation when inference fails. |
| 5 | Report Generation | Produces a structured PDF with condition, differential, recommendation, watch-for list, urgency level, confidence badge (Green/Yellow/Red), cascade audit trail, and RAG evidence citations. |

## Safety Layer

16 hard red-flag patterns run before any LLM query. Patterns include:

| Flag Class | Key Criteria | Min Match |
|-----------|-------------|-----------|
| Suicidal Crisis | Suicidal ideation | 1 |
| Stroke (FAST) | Facial droop, speech difficulty, unilateral weakness | 1 |
| ACS/MI | Chest pain/arm pain with sweating, dyspnea, or syncope | 2 |
| Anaphylaxis | Face/throat swelling, stridor, SOB, rash | 2 |
| Seizure | Reported seizure or convulsion | 1 |
| GI Bleed | Hematemesis, melena, or rectal bleeding | 1 |
| Meningitis | Neck stiffness or photophobia with fever | 2 |
| SAH (Thunderclap) | Sudden severe headache | 1 |
| Pulmonary Embolism | SOB/chest pain with DVT signs or hemoptysis | 2 |
| Respiratory Emergency | Severe dyspnea, stridor, or cyanosis | 2 |
| Head Trauma | Head injury with LOC, vomiting, or confusion | 2 |
| DKA/HHS | Polyuria/polydipsia with systemic signs | 3 |
| Sepsis | Fever with multi-system signs | 3 |
| Infant Fever | Any fever under 3 months | 1 |
| Sick Infant | Poor feeding, lethargy, or cyanosis under 12 months | 1 |
| Severe Dehydration | Vomiting/diarrhea with volume-depletion signs | 3 |
| Obstetric Emergency | Bleeding, severe pain, or syncope in pregnancy | 2 |

Key design decisions:
- Stroke uses FAST-style neuro keys only (not generic headache plus dizziness)
- Sepsis requires minimum 3 matches AND fever to reduce false positives
- Age and pregnancy gates for infant and obstetric patterns
- Escalation bias applied when confidence drops below 0.65
- Conservative confidence fusion: prefer the lower of scorer and model confidence when they disagree

## Models

| Model | Use | Hosting |
|-------|-----|---------|
| Whisper Large V3 Turbo | Conversational Urdu/Hindi/English transcription | Local CPU INT8 or NVIDIA CUDA via faster-whisper |
| OpenRouter free router | Triage decision-support inference | OpenRouter (`openrouter/free`; model availability varies) |
| EmbeddingGemma-300M Medical | Optional RAG embeddings | Local CPU/GPU via sentence-transformers |

## Tech Stack

**Backend:**
- FastAPI with SSE streaming endpoints
- LangChain agent framework with deterministic routing
- ChromaDB vector store for medical guidelines RAG
- WeasyPrint for PDF report generation
- faster-whisper for local multilingual ASR
- Sentence-Transformers for medical embeddings
- uroman + Indic transliteration for Roman Urdu/Hindi normalization

**Frontend:**
- React 19 with TypeScript
- Vite 8 build tooling
- Voice capture via browser MediaRecorder API with PCM/WAV encoding

## Prerequisites

### Required for full pipeline:
- **Python 3.11+** with uv package manager
- **Whisper Large V3 Turbo** (one-time local download; CPU INT8 supported, NVIDIA CUDA optional)
- **OpenRouter API key** for live model inference (the default `openrouter/free` model is for testing)
- **Optional CPU/GPU resources** for ChromaDB + EmbeddingGemma medical RAG embeddings
- **Optional remote ASR server** (only when not using local Whisper)
- **WHO ICD-11 API key** (optional, for RAG corpus loading)

### For testing without GPU:
Parser, safety, scorer, seed retrieval, cascade heuristics, PDF, and local voice ASR work offline after the Whisper model is cached. Live model inference requires OpenRouter network access.

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/imadnan4/MedRoute.git
cd MedRoute

# Copy and fill in your configuration
cp .env.example backend/.env
# Edit backend/.env with your API keys (see Configuration below)
```

### 2. Backend

```bash
cd backend

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# One-time: download local Whisper Large V3 Turbo
python scripts/download_asr_model.py

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### 3. Frontend (optional, for the web UI)

```bash
cd frontend
npm install
npm run dev
```

The dev server proxies API calls to the backend on port 8000. The production build in `frontend/dist` is served directly by FastAPI.

### 4. Docker

```bash
cp .env.example .env
# Fill in your API keys in .env
docker compose up --build
```

## Configuration

All configuration lives in `backend/.env`. Copy from `.env.example` and fill in:

| Variable | Required | Description |
|----------|----------|-------------|
| `MEDROUTE_ASR_MODE` | No (default `auto`) | `auto` / `local` / `remote` -- prefers local ONNX when model is cached |
| `MEDROUTE_ASR_SERVER_URL` | Only if `remote` | Optional NeMo server (e.g. `http://AMD_IP:8080`) |
| `MEDROUTE_OPENROUTER_API_KEY` | For live inference | API key from openrouter.ai |
| `MEDROUTE_OPENROUTER_MODEL` | No (default `openrouter/free`) | OpenRouter model ID; use a specific `:free` model when reproducibility matters |
| `MEDROUTE_ICD_API_KEY` | For RAG corpus | Free token from id.who.int |
| `MEDROUTE_HF_TOKEN` | Optional | HuggingFace token for faster model downloads |

### Setting up OpenRouter

1. Create an API key at https://openrouter.ai/settings/keys.
2. Set `MEDROUTE_OPENROUTER_API_KEY` in `backend/.env`.
3. Keep `MEDROUTE_OPENROUTER_MODEL=openrouter/free` for testing, or pin a specific `:free` model for reproducible evaluation.

Do not send real patient-identifiable data through free hosted models. Free-model providers and availability can change, and their data-handling policies may differ.


### Loading the RAG Corpus (optional)

```bash
cd backend
source .venv/bin/activate
python -c "from rag.loader import load_all; load_all()"
```

This downloads WHO ICD-11 definitions and Medical Meadow WikiDoc into ChromaDB. RAG is retained because it gives the model a small set of relevant, inspectable guideline passages instead of asking it to rely only on model memory. The built-in seed retriever remains available when ChromaDB is absent.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/triage` | Run full triage pipeline (accepts text or base64 audio) |
| GET | `/triage/stream` | SSE streaming pipeline stages with progress events |
| POST | `/transcribe` | Transcribe audio via local Whisper or a configured remote ASR server |
| GET | `/report/{case_id}` | Download PDF report for a previously run triage |
| GET | `/health` | Health check with version info |

### Example: POST /triage

```bash
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Mujhe do din se zukam, halka bukhar aur sar dard hai",
    "language": "hi-IN",
    "age_years": 28
  }'
```

### Example: POST /transcribe

```bash
# Transcribe base64-encoded audio
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio_b64": "<base64-encoded-wav>",
    "language": "ur"
  }'
```

### Example: GET /triage/stream (SSE)

```bash
curl -N "http://localhost:8000/triage/stream?transcript=Mujhe+bukhar+hai&language=hi-IN&age_years=28"
```

Events stream as: `asr -> parser -> safety -> scorer -> agent -> done`

## Demo Scenarios

**A -- Simple Case (OpenRouter Free Model)**
Input: "Mujhe do din se zukam, halka bukhar aur sar dard hai" (28yr, not pregnant)
Expected: Direct OpenRouter inference, GREEN confidence, viral URI

**B -- Red Flag (Hard Override)**
Input: "Chest tightness, left arm pain, sweating" (58yr male)
Expected: Hard escalation, EMERGENCY urgency

**C -- Complex Case (OpenRouter + RAG)**
Input: "Fatigue 3 weeks, 5kg weight loss, night sweats" (42yr male)
Expected: RAG-grounded OpenRouter inference, YELLOW confidence, lymphoma/TB differential

**D -- Low Confidence (Escalation Bias)**
Input: "I feel off. Tired. Something is wrong." (45yr male)
Expected: Escalation bias applied, RED confidence badge, clinician referral

## ASR Modes

| Mode | Behavior |
|------|----------|
| `auto` (default) | Use local Whisper when faster-whisper is installed, otherwise try the remote URL |
| `local` | Force local Whisper (`~/.cache/medroute/whisper`) |
| `remote` | Force the HTTP ASR server at `MEDROUTE_ASR_SERVER_URL` |

One-time model download:
```bash
cd backend
.venv/bin/python scripts/download_asr_model.py
```

The frontend captures native-rate mono PCM/WAV and lets Whisper perform high-quality resampling. Empty transcript results return a soft response so the user can retry or edit the text manually.

## Project Structure

```
.
├── backend
│   ├── agents
│   │   ├── tools
│   │   │   ├── escalate_uncertain.py
│   │   │   ├── __init__.py
│   │   │   ├── openrouter_infer.py
│   │   │   └── rag_search.py
│   │   ├── clinical_heuristics.py
│   │   ├── __init__.py
│   │   └── triage_agent.py
│   ├── pipeline
│   │   ├── templates
│   │   │   └── report.html
│   │   ├── complexity_scorer.py
│   │   ├── __init__.py
│   │   ├── input_parser.py
│   │   ├── report_generator.py
│   │   └── triage_pipeline.py
│   ├── rag
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── retriever.py
│   ├── safety
│   │   ├── __init__.py
│   │   └── red_flag_checker.py
│   ├── scripts
│   │   └── download_asr_model.py
│   ├── voice
│   │   ├── __init__.py
│   │   ├── roman_urdu.py
│   │   ├── transcriber.py
│   │   └── whisper_local.py
│   ├── clinical_knowledge.py
│   ├── config.py
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── frontend
│   ├── public
│   │   ├── favicon.svg
│   │   └── icons.svg
│   ├── src
│   │   ├── components
│   │   │   ├── ReportDownload.tsx
│   │   │   ├── TriageResult.tsx
│   │   │   └── VoiceRecorder.tsx
│   │   ├── api.ts
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   └── types.ts
│   ├── .gitignore
│   ├── index.html
│   ├── .oxlintrc.json
│   ├── package.json
│   ├── package-lock.json
│   ├── README.md
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .gitignore
├── mise.toml
└── README.md
```

## Pipeline architecture

The full triage flow lives in one **deep module**, `backend/pipeline/triage_pipeline.py`,
which owns the stage order (`asr → parser → safety → scorer → agent`) and assembles the
structured outcome that the web layer previously hand-built in two places. Both HTTP
endpoints (`POST /triage` and `GET /triage/stream`) call a single `TriagePipeline.run(...)`
interface; the SSE endpoint passes an `on_stage(name, status, payload)` callback to observe
progress without the pipeline knowing anything about SSE. Stage dependencies are injected at
construction, so the module is testable with fakes (`backend/tests/test_triage_pipeline.py`,
run offline via `pytest`). This concentrates the orchestration in one place and keeps the web
layer free of pipeline state.

The LLM inference call is behind an `InferenceAdapter` seam (`backend/agents/tools/inference.py`):
`OpenRouterInfer` is the production implementation, `InMemoryInfer` a test double, so the
triage cascade is exercisable offline. The report generator (`backend/pipeline/report_generator.py`)
loads its template lazily and accepts an injected `now=`, making PDF output deterministic and testable.
The dead network-coupled RAG `loader.py` was removed; retrieval runs on seed knowledge.
All shared clinical knowledge (symptom lexicon, syndrome clusters, cluster bonuses, red-flag
patterns) lives in one owned module, `backend/clinical_knowledge.py`, so the parser, scorer,
and safety checker read from a single source of truth instead of leaking copies across seams.
The backend test suite (`backend/tests/`, run with `pytest`) covers the pipeline, inference
adapter, report generator, and retriever offline.

## Build Progress

What works offline (no GPU):
- Full pipeline: parser -> safety -> scorer -> deterministic orchestrator -> PDF
- Demo scenarios retain deterministic routing/safety and heuristic fallback without model access
- Seed RAG knowledge when ChromaDB empty or missing
- Cascade: OpenRouter failure -> clinical heuristics -> clinician escalation
- Confidence fusion + urgency + cascade audit trail
- Local Whisper Large V3 Turbo ASR with native-rate browser PCM/WAV capture
- FastAPI endpoints: /triage, /triage/stream, /transcribe, /report/{id}, /health

What needs external services/resources:
- OpenRouter API access for live model inference
- ChromaDB + EmbeddingGemma for full vector RAG (seed retrieval still works without them)

## Design Decisions

| Decision | Reason |
|----------|--------|
| Deterministic orchestrator over pure ReAct | ReAct ignores route hints; unsafe for triage |
| Seed RAG + clinical heuristics | Offline quality without GPU |
| Confidence fusion: min when disagree | Medical LLM uncertainty literature |
| FAST stroke keys in lexicon | Prior stroke pattern too nonspecific |
| Sepsis min_match=3 + require fever | Reduce false positives on mild URI |
| uv package manager | Fast, deterministic Python dependency management |
| Local Whisper ASR | Native Urdu support and strong conversational/code-switching recognition |
| Native-rate PCM/WAV capture | Preserves audio quality and avoids MediaRecorder/WebM decode inconsistencies |

## Deployment

Config-only setup for **MedRoute**. The backend (FastAPI) deploys to **Heroku**; the frontend (React + Vite) deploys to **Vercel**. No real patient data is used — MedRoute ships with bundled **sample/synthetic transcripts** for local testing and runs on that data out of the box, not a production medical device.

### Backend → Heroku

The backend ships `backend/Procfile` and `backend/runtime.txt` (Python 3.12.7). The Procfile runs:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Heroku sets `$PORT`; do not hardcode a port. Deploy the `backend/` directory as its own Heroku app:

```bash
cd backend

# Create the app (Heroku detects the Python buildpack from runtime.txt)
heroku create medroute-app

# Required env vars (see .env.example)
heroku config:set MEDROUTE_OPENROUTER_API_KEY=sk-or-v1-...
heroku config:set MEDROUTE_ASR_MODE=groq
# Optional:
heroku config:set MEDROUTE_GROQ_API_KEY=gsk_...
heroku config:set MEDROUTE_ICD_API_KEY=       # optional, RAG corpus

# Push
git subtree push --prefix backend heroku main
# or, with a backend-only remote: git push heroku main
```

`requirements.txt` already includes `fastapi`, `uvicorn`, `pydantic`, `httpx`, and `requests` — all build deps are satisfied.

> **ASR note:** hosted ASR uses **Groq Whisper** (`MEDROUTE_ASR_MODE=groq`), which needs `MEDROUTE_GROQ_API_KEY`, because Heroku has no GPU. Set `MEDROUTE_ASR_MODE=groq` so transcription runs without a cached local model.

### Frontend → Netlify

The frontend is deployed at `https://your-production-frontend.example.com`. Set these
production environment variables in the Netlify site settings:

- `VITE_API_URL` = `https://medroute-api.herokuapp.com`
- `VITE_NEON_AUTH_URL` = the Neon Auth URL from `frontend/.env.example`

For a manual site configuration, use `frontend` as the base directory, run
`npm run build`, and publish `dist`.

`VITE_API_URL` is read in `frontend/src/api.ts` (`import.meta.env.VITE_API_URL`). Without it, the UI falls back to `http://localhost:8000`.

### Neon Auth and durable storage

MedRoute uses Neon Auth (managed Better Auth) for browser sign-in and stores each
Encounter, its structured result, submitted audio, and generated PDF in Neon
Postgres. The FastAPI API accepts only Neon Auth JWTs and scopes every stored
Encounter by the JWT `sub` claim.

1. Enable Neon Auth for the database branch and add the deployed frontend URL to
   Neon Auth's trusted origins.
2. Copy the environment templates and set:

   ```bash
   cp .env.example .env
   cp frontend/.env.example frontend/.env.local
   ```

   Set `MEDROUTE_DATABASE_URL` to the pooled connection URL for the API,
   `MEDROUTE_DATABASE_URL_UNPOOLED` to the direct URL for migrations,
   `MEDROUTE_AUTH_JWKS_URL` to the branch JWKS URL, and
   `VITE_NEON_AUTH_URL` to the branch Auth URL. Do not commit either env file.

3. Apply the checked-in schema with the direct URL:

   ```bash
   MEDROUTE_DATABASE_URL_UNPOOLED="...direct Neon URL..." \
     ./.venv/bin/python scripts/migrate_neon.py
   ```

Neon Auth owns its `neon_auth` schema; `migrations/001_medroute_encounters.sql`
only creates the application table. The API deliberately uses server-side SQL
instead of the optional Neon Data API because this app already has a FastAPI
backend and needs to persist binary report/audio data in the same Postgres
transaction boundary.

### Required environment variables (real names from `backend/config.py`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `MEDROUTE_OPENROUTER_API_KEY` | Yes | LLM triage inference via OpenRouter |
| `MEDROUTE_ASR_MODE` | Yes (set `groq`) | Hosted Groq Whisper ASR — no local GPU needed |
| `MEDROUTE_GROQ_API_KEY` | Yes (when `asr_mode=groq`) | Groq API key for hosted Whisper ASR |
| `MEDROUTE_ICD_API_KEY` | Optional | WHO ICD-11 RAG corpus loading |
| `MEDROUTE_OPENROUTER_MODEL` | No (default `openrouter/free`) | OpenRouter model ID |
| `MEDROUTE_DATABASE_URL` | Yes | Pooled Neon Postgres URL for API traffic |
| `MEDROUTE_DATABASE_URL_UNPOOLED` | Yes for migrations | Direct Neon Postgres URL |
| `MEDROUTE_AUTH_JWKS_URL` | Yes | Neon Auth JWKS endpoint used to verify bearer tokens |
| `MEDROUTE_CORS_ORIGINS` | Yes in deployment | Comma-separated frontend origins |

Never commit `backend/.env` or any secret. Use `heroku config:set` / Netlify environment variables instead.

## License

MIT

Built for AMD Developer Hackathon ACT II by Adnan Ahmad
