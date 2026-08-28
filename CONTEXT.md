# MedRoute Domain Model

MedRoute is a healthcare first-contact engine. It turns a raw patient presentation (voice or text) into a structured record, assesses acuity, and decides a disposition. **Intake** and **Triage** are two modes over the same core — they are not separate products.

## First contact

**Encounter**
A single first-contact event — one call or message — that yields one Presentation.
_Avoid_: session, case, ticket.

**Presentation**
The content a person communicates at first contact: symptoms, complaints, or care needs. May arrive as typed text or as a spoken account later rendered as a Transcript.
_Avoid_: input, message.

**Transcript**
The verbatim text produced by speech recognition from a spoken Presentation. Distinct from the Presentation, which is the interpreted account.
_Avoid_: input.

**Speaker**
The individual who provides the Presentation. May be the Person, or a proxy such as a caregiver.
_Avoid_: caller.

**Intake Record**
The structured capture of an Encounter's Presentation (who, what they need, context), produced by the Intake mode.
_Avoid_: form, record.

## The person

**Person**
The individual whose care is being discussed — the patient, care-recipient, or client. Distinct from the Speaker when a caregiver speaks on their behalf.
_Avoid_: patient, client, account.

## Assessment

**Red Flag**
A deterministic, pre-LLM hard-stop condition (e.g. suicidal ideation, stroke signs) that forces immediate escalation regardless of Acuity. Evaluated before any model call.
_Avoid_: safety flag.

**Acuity**
A categorical severity derived from the Presentation: `routine`, `priority`, or `urgent`.
_Avoid_: urgency, triage level.

**Clinical Knowledge**
The single owned body of symptom, syndrome, and red-flag data used to detect Red Flags and derive Acuity. Both modes read it; no copy lives outside the one owned source.
_Avoid_: lexicon, ruleset (those are implementation words for the same idea).

## Decision

**Disposition**
The decision of where an Encounter goes next: `escalate_to_clinician` (human now), `standard_queue` (routine intake), or `provide_guidance` (self-care or clinical guidance without a live human).
_Avoid_: routing, route.

**Mode**
The entry funnel applied to an Encounter. `Intake` captures the structured record and queues it; `Triage` assesses Acuity and may provide clinical guidance. Both share the same Red-Flag and Acuity core.
_Avoid_: vertical, workflow.
