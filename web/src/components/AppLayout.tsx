/**
 * The frame after login (docs/ARCHITECTURE.md §10): a top bar with the app, the family, the member, the role and a
 * logout button, and the navigation. Only the chat exists in step 9a; the other pages are shown greyed out as "قريباً".
 * It loads the profile itself (decision A).
 */
import { useEffect, useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router";

import { getMe, type Profile } from "../api/client";
import { clearToken } from "../auth/session";

const SOON = ["لوحة الشهر", "المصاريف", "العائلة"];

export default function AppLayout() {
  const navigate = useNavigate();
  const [profile, setProfile] = useState<Profile | null>(null);

  useEffect(() => {
    let active = true;
    // A failure here is a 401 (handled by the client) or a server that is down, which the chat page reports too.
    getMe().then((found) => active && setProfile(found)).catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  function logout() {
    clearToken();
    navigate("/login", { replace: true });
  }

  const role = profile?.member.role;
  return (
    <div className="flex h-dvh flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-3xl flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3">
          <span className="text-2xl font-bold text-teal-700">ميزان</span>
          {profile && (
            <>
              <span className="text-sm text-slate-500">{profile.family.name}</span>
              <span className="ms-auto text-sm font-semibold">{profile.member.name}</span>
              <span
                className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                  role === "manager" ? "bg-teal-100 text-teal-800" : "bg-slate-100 text-slate-700"
                }`}
              >
                {role === "manager" ? "مسؤول" : "فرد"}
              </span>
            </>
          )}
          <button
            type="button"
            onClick={logout}
            className={`${profile ? "" : "ms-auto "}rounded-lg px-3 py-1 text-sm text-slate-600 ring-1 ring-slate-300 hover:bg-slate-100`}
          >
            خروج
          </button>
        </div>
        <nav className="mx-auto flex max-w-3xl gap-1 overflow-x-auto px-4 pb-2 text-sm" aria-label="التنقّل">
          <NavLink
            to="/chat"
            className={({ isActive }) =>
              `whitespace-nowrap rounded-lg px-3 py-1.5 font-semibold ${isActive ? "bg-teal-700 text-white" : "text-teal-700"}`
            }
          >
            المحادثة
          </NavLink>
          {SOON.map((name) => (
            <span key={name} aria-disabled="true" className="cursor-not-allowed whitespace-nowrap rounded-lg px-3 py-1.5 text-slate-400">
              {name} <span className="text-xs">(قريباً)</span>
            </span>
          ))}
        </nav>
      </header>
      <div className="min-h-0 flex-1">
        <Outlet />
      </div>
    </div>
  );
}
