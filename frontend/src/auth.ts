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

// @neondatabase/auth 0.5.0-beta infers the React adapter's useSession member
// as an atom even though the runtime React client exposes useSession() as a
// hook. Keep that package typing mismatch isolated at this integration seam.
export const useAuthSession = authClient.useSession as unknown as () => {
  data: { user: { name?: string | null; email?: string | null } } | null;
  isPending: boolean;
};
