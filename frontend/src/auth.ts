import { createInternalNeonAuth } from "@neondatabase/auth";
import { BetterAuthReactAdapter } from "@neondatabase/auth/react/adapters";

const authUrl =
  import.meta.env.VITE_NEON_AUTH_URL || "http://localhost:8000/__auth_not_configured";

const neonAuth = createInternalNeonAuth(authUrl, {
  adapter: BetterAuthReactAdapter(),
});

export const authClient = neonAuth.adapter;
export const getAuthToken = neonAuth.getJWTToken;
export const isAuthConfigured = Boolean(import.meta.env.VITE_NEON_AUTH_URL);
