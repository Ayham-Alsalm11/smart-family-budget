/** Pages behind the login (docs/ARCHITECTURE.md §10): without a token the member goes to the login page. */
import { Navigate, Outlet } from "react-router";

import { getToken } from "./session";

export default function RequireAuth() {
  return getToken() ? <Outlet /> : <Navigate to="/login" replace />;
}
