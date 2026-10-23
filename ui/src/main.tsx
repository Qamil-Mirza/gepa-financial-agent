import "@fontsource/instrument-sans/400.css";
import "@fontsource/instrument-sans/500.css";
import "@fontsource/instrument-sans/600.css";
import "@fontsource/instrument-sans/700.css";
import "@fontsource/newsreader/500.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/pitwall.css";
import "./styles/workflow.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { applyTheme, readTheme } from "./components/ThemeToggle";

applyTheme(readTheme()); // before the first render, so a pinned theme never flashes the system one

const root = document.getElementById("root");
if (!root) throw new Error("index.html has no #root");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
