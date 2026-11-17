/** Vitest setup (docs/ARCHITECTURE.md §10): DOM matchers, and a clean DOM and localStorage after every test. */
import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
