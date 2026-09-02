-- MedRoute application data. Neon Auth owns the neon_auth schema.
CREATE TABLE IF NOT EXISTS medroute_encounters (
    case_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('triage', 'intake')),
    request JSONB NOT NULL DEFAULT '{}'::jsonb,
    stages JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    audio_bytes BYTEA,
    pdf_bytes BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS medroute_encounters_user_created_idx
    ON medroute_encounters (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS medroute_encounters_result_gin_idx
    ON medroute_encounters USING GIN (result);
