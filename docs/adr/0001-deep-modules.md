# ADR-0001: Deep-module consolidation of the triage backend

## Status

Accepted — implemented 2026-08-28.

## Context

An architecture review (`/improve-codebase-architecture`) found the backend was shallow in
several places: the pipeline stage order was duplicated in the web layer (`main.py`
`run_pipeline` + the SSE `event_stream`), the LLM inference call had no swappable seam, the
report generator was non-deterministic, a dead network-coupled RAG loader shipped, and the
clinical-knowledge data (symptom lexicon, syndrome clusters, cluster bonuses, red-flag
patterns) was copied across `input_parser`, `complexity_scorer`, and `red_flag_checker`.

## Decision

Consolidate each friction point into a deep module — a small interface with the behaviour
behind it, so callers and tests cross one seam:

1. **`backend/pipeline/triage_pipeline.py`** — `TriagePipeline.run(request, case_id, *, on_stage=None)`
   owns the full stage order (`asr → parser → safety → scorer → agent`) and the assembled
   outcome. Both `/triage` and `/triage/stream` leverage it; SSE observes progress via the
   `on_stage` callback. Dependencies injected at construction (testable, no global state).
2. **`backend/agents/tools/inference.py`** — `InferenceAdapter` seam with `OpenRouterInfer`
   (prod) and `InMemoryInfer` (test double); `triage_agent.run_triage` takes the adapter.
3. **`backend/pipeline/report_generator.py`** — template loads lazily; `now=` injected, so
   PDF output is deterministic and testable.
4. **`backend/clinical_knowledge.py`** — single owned source of truth for `SYMPTOM_LEXICON`,
   `SYNDROME_CLUSTERS`, `CLUSTER_BONUS`, `RED_FLAG_PATTERNS`, plus shared `has_symptom`. The
   parser, scorer, and safety checker import from it; no leaked copies.

The dead `backend/rag/loader.py` was deleted; retrieval runs on `SEED_KNOWLEDGE`.

## Consequences

- Locality: each concept now lives in one module (pipeline order, inference, clinical
  knowledge, report rendering).
- Leverage: one interface, N call sites; tests exercise the seam, not internals.
- The four deep modules are covered by offline `pytest` suites under `backend/tests/`.

## Non-goals / deferred

- The clinical-knowledge merge moves the DATA verbatim and unifies the alias-matching helper,
  but does NOT reconcile divergent matching semantics between `red_flag_checker` (e.g. meningitis
  requires `neck_stiffness` OR `photophobia` AND ≥2 symptoms) and `clinical_heuristics`
  (looser OR). Revisiting those semantics is a clinical-decision change, out of scope for this
  structural ADR — do not "fix" it as a refactor.
- No new external seams (e.g. real local-LLM adapter, wired corpus loader) were introduced;
  those remain hypothetical until a second real variation appears.
