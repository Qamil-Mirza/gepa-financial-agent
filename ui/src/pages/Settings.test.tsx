import { render, screen, within } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Health, Settings } from "../api/client";
import { workflowHref } from "../routes";
import { fetchStub, health, settings } from "../test/fixtures";
import { SettingsPage } from "./Settings";

const CONFIG_NOTE = "Edit ~/.pitstop/config.toml, see docs/config.example.toml";

/** The fixture with what it lacks: a global budget, a judge key that is set, Slack configured with its env var
 *  set, email configured without it, and a running worker. */
const full: Settings = {
  ...settings,
  budgets: [
    { name: "crusoe", scope: "global:crusoe", period: "total", spent_usd: 12.5, limit_usd: 50, remaining_usd: 37.5 },
  ],
  judge: { ...settings.judge, key_set: true },
  alerts: {
    ...settings.alerts,
    channels: settings.alerts.channels.map((c) =>
      c.name === "slack" ? { ...c, configured: true, env_set: true } : c.name === "email" ? { ...c, configured: true } : c,
    ),
  },
};

const fullHealth: Health = {
  ...health,
  worker: { enabled: true, running: true, last_scan_at: 1_792_000_000, last_scan_runs: 3, last_error: null, scan_every_s: 60 },
};

function renderPage(body: Settings = full, h: Health = fullHealth) {
  vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/settings": body, "/api/health": h })));
  const router = createMemoryRouter(
    [
      { path: "/settings", element: <SettingsPage /> },
      { path: "/w", element: <p>workflow page</p> },
    ],
    { initialEntries: ["/settings"] },
  );
  render(<RouterProvider router={router} />);
}

/** The cell after the row header `label` in a table, or the value after a term in a definition list. */
function rowOf(region: HTMLElement, name: string): HTMLElement {
  const row = within(region).getByRole("rowheader", { name }).closest("tr");
  if (!(row instanceof HTMLElement)) throw new Error(`no row for ${name}`);
  return row;
}

function valueOf(region: HTMLElement, term: string): HTMLElement {
  const dd = within(region).getByText(term, { selector: "dt" }).nextElementSibling;
  if (!(dd instanceof HTMLElement)) throw new Error(`no value for ${term}`);
  return dd;
}

describe("SettingsPage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("renders each section from fixture", async () => {
    renderPage();
    // workflows: one row each, linked to its page, with a note that they are set in code
    const workflows = await screen.findByRole("region", { name: "Workflows" });
    expect(screen.getByRole("heading", { level: 1, name: "Settings" })).toBeVisible();
    expect(within(workflows).getByRole("table", { name: "Workflows" })).toBeInTheDocument();
    expect(within(workflows).getAllByRole("row")).toHaveLength(1 + settings.workflows.length);
    expect(within(workflows).getByRole("link", { name: "earnings-research" })).toHaveAttribute(
      "href",
      workflowHref("earnings-research"),
    );
    const earnings = rowOf(workflows, "earnings-research");
    expect(within(earnings).getAllByRole("cell").map((c) => c.textContent)).toEqual(["high", "normal", "manual", "$2.00"]);
    expect(within(rowOf(workflows, "nightly-digest")).getAllByRole("cell")[3]).toHaveTextContent("none");
    expect(workflows).toHaveTextContent("Set in code: pitstop.workflow(..., criticality=…)");
    expect(workflows).not.toHaveTextContent(CONFIG_NOTE);

    // global budgets: spent and limit
    const budgets = screen.getByRole("region", { name: "Budgets" });
    const crusoe = rowOf(budgets, "crusoe");
    expect(crusoe).toHaveTextContent("all time");
    expect(crusoe).toHaveTextContent("$12.50");
    expect(crusoe).toHaveTextContent("$50.00");
    expect(budgets).toHaveTextContent(CONFIG_NOTE);

    // the judge: model, endpoint, the key's env var by name and whether it is set
    const judge = screen.getByRole("region", { name: "Pre-action judge" });
    expect(valueOf(judge, "Model")).toHaveTextContent("openai/gpt-oss-120b");
    expect(valueOf(judge, "Base URL")).toHaveTextContent("https://api.inference.crusoecloud.com/v1");
    expect(valueOf(judge, "API key")).toHaveTextContent(/^CRUSOE_API_KEY set$/);
    expect(judge).toHaveTextContent(CONFIG_NOTE);

    // alert channels and the routes per level
    const alerts = screen.getByRole("region", { name: "Alert channels" });
    expect(within(alerts).getAllByRole("row")).toHaveLength(1 + settings.alerts.channels.length);
    const slack = rowOf(alerts, "slack");
    expect(within(slack).getAllByRole("cell").map((c) => c.textContent)).toEqual(["yes", "SLACK_WEBHOOK_URL", "set"]);
    expect(within(rowOf(alerts, "pagerduty")).getAllByRole("cell").map((c) => c.textContent)).toEqual([
      "no",
      "PAGERDUTY_ROUTING_KEY",
      "not set",
    ]);
    expect(rowOf(alerts, "email")).toHaveTextContent("not set");
    const routes = within(alerts).getByRole("list", { name: "Alert routes" });
    const items = within(routes).getAllByRole("listitem");
    expect(items.map((li) => li.querySelector(".route-level")?.textContent)).toEqual(["P0", "P1", "Digest (P2, P3)"]);
    expect(items[0]).toHaveTextContent("pagerduty, opsgenie, slack, webhook");
    expect(items[0]).toHaveTextContent("Sends to slack");
    expect(items[1]).toHaveTextContent("Sends to slack");
    expect(alerts).toHaveTextContent(CONFIG_NOTE);

    // the background worker, from /api/health
    const worker = screen.getByRole("region", { name: "Background worker" });
    expect(valueOf(worker, "Status")).toHaveTextContent("Running");
    expect(valueOf(worker, "Scans every")).toHaveTextContent("1 min");
    expect(valueOf(worker, "Last scan")).toHaveTextContent("3 runs");
    expect(valueOf(worker, "Last error")).toHaveTextContent("none");

    // where data lives
    const home = screen.getByRole("region", { name: "Where data lives" });
    expect(valueOf(home, "PITSTOP_HOME")).toHaveTextContent("/home/you/.pitstop");
    expect(home).toHaveTextContent("/home/you/.pitstop/config.toml");
  });

  it("shows not set for missing keys", async () => {
    renderPage(settings, health);
    const judge = await screen.findByRole("region", { name: "Pre-action judge" });
    expect(valueOf(judge, "API key")).toHaveTextContent(/^CRUSOE_API_KEY not set$/);
    expect(judge).toHaveTextContent("judge_client");

    const alerts = screen.getByRole("region", { name: "Alert channels" });
    for (const c of settings.alerts.channels) {
      const cells = within(rowOf(alerts, c.name)).getAllByRole("cell");
      expect(cells[1]).toHaveTextContent(c.env_var);
      expect(cells[2]).toHaveTextContent(/^not set$/);
    }
    const routes = within(alerts).getByRole("list", { name: "Alert routes" });
    for (const li of within(routes).getAllByRole("listitem")) expect(li).toHaveTextContent("No channel here is ready");

    // no global budget yet; the worker is off under --no-worker
    expect(screen.getByRole("region", { name: "Budgets" })).toHaveTextContent("No global budget");
    const worker = screen.getByRole("region", { name: "Background worker" });
    expect(valueOf(worker, "Status")).toHaveTextContent("Off");
    expect(valueOf(worker, "Last scan")).toHaveTextContent("none yet");
  });

  it("announces loading and offers a retry when settings fail", async () => {
    vi.stubGlobal("fetch", vi.fn(fetchStub({ "/api/health": health })));
    const router = createMemoryRouter([{ path: "/settings", element: <SettingsPage /> }], { initialEntries: ["/settings"] });
    render(<RouterProvider router={router} />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading settings");
    expect(await screen.findByRole("alert")).toHaveTextContent("Settings didn't load");
    expect(screen.getByRole("button", { name: "Try again" })).toBeVisible();
  });
});
