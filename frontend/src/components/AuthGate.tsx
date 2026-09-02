import { useState, type ReactNode } from "react";
import { authClient, isAuthConfigured, useAuthSession } from "../auth";

interface AuthGateProps {
  children: ReactNode;
}

function responseError(response: { error?: { message?: string } | null }) {
  return response.error?.message || "Authentication failed. Please try again.";
}

function AuthForm() {
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleGoogleSignIn() {
    setSubmitting(true);
    setError("");
    try {
      const response = await authClient.signIn.social({
        provider: "google",
        callbackURL: window.location.origin,
      });
      if (response.error) setError(responseError(response));
    } catch (caught: unknown) {
      setError(caught instanceof Error ? caught.message : "Authentication failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <p className="eyebrow">Private clinical workspace</p>
        <h1 id="auth-title">Sign in to MedRoute</h1>
        <p className="auth-copy">
          Your first-contact records and reports are stored in your Neon database
          and only available to your account.
        </p>
        <div className="auth-form">
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="btn btn-primary btn-large" type="button"
            onClick={() => void handleGoogleSignIn()} disabled={submitting}>
            {submitting ? "Connecting to Google…" : "Continue with Google"}
          </button>
          <p className="auth-copy auth-provider-note">
            Sign in or create your account securely with Google.
          </p>
        </div>
      </section>
    </main>
  );
}

export default function AuthGate({ children }: AuthGateProps) {
  const session = useAuthSession();

  if (!isAuthConfigured) {
    return (
      <main className="auth-page">
        <section className="auth-card" aria-labelledby="auth-config-title">
          <p className="eyebrow">Setup required</p>
          <h1 id="auth-config-title">Connect Neon Auth</h1>
          <p className="auth-copy">
            Add VITE_NEON_AUTH_URL to the frontend environment before signing
            in. The API also needs the matching Neon JWKS URL.
          </p>
        </section>
      </main>
    );
  }

  if (session.isPending) {
    return <main className="auth-page"><p className="auth-loading">Checking your session…</p></main>;
  }
  if (!session.data?.user) return <AuthForm />;

  return (
    <>
      <div className="auth-toolbar">
        <span>Signed in as <strong>{session.data.user.name || session.data.user.email}</strong></span>
        <button type="button" onClick={() => void authClient.signOut()}>Sign out</button>
      </div>
      {children}
    </>
  );
}
