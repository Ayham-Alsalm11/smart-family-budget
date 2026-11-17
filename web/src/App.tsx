/**
 * The routes of the web interface (docs/ARCHITECTURE.md §10). The account pages are public; everything else needs
 * the token. When the client sees a session end (401), it calls the handler set here, which opens the login page.
 */
import { useEffect } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router";

import { setUnauthorizedHandler } from "./api/client";
import RequireAuth from "./auth/RequireAuth";
import AppLayout from "./components/AppLayout";
import ChatPage from "./pages/ChatPage";
import JoinPage from "./pages/JoinPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";

export default function App() {
  const navigate = useNavigate();

  useEffect(() => {
    setUnauthorizedHandler(() => navigate("/login", { replace: true }));
  }, [navigate]);

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/join" element={<JoinPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppLayout />}>
          <Route path="/chat" element={<ChatPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}
