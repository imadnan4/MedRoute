# Unified intake + triage engine: one core, two modes

MedRoute was originally a medical-triage agent. We are expanding it into a healthcare first-contact platform where **Intake** and **Triage** are two entry modes over one shared engine, not two products.

The shared core — deterministic Red-Flag detection, Acuity scoring, the owned Clinical Knowledge, the ASR and inference seams, and the orchestration pipeline — is built once and leveraged by both modes. Intake and Triage each contribute only a thin adapter at the seam (extraction schema, Disposition vocabulary). A single `Person` timeline is the single source of truth: the Intake Record is the input to Triage, and the Triage assessment enriches the same Encounter.

This reverses an earlier plan to build a separate intake-only fork of MedRoute. Building on top avoids duplicating the deep modules we just consolidated and keeps one test suite and one deployable.
