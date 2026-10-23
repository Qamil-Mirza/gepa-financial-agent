// Typed views of the JSON fixtures (real API responses on the demo data; regenerate with
// `uv run python ui/scripts/gen_fixtures.py`). JSON imports widen literal types, hence the casts.
import type {
  Board,
  Changes,
  Explain,
  FeedItem,
  Graph,
  Health,
  RunDetail,
  RunPage,
  Settings,
  WorkflowDetail,
} from "../api/client";
import boardJson from "./fixtures/board.json";
import boardEmptyJson from "./fixtures/board_empty.json";
import changesJson from "./fixtures/changes.json";
import explainJson from "./fixtures/explain.json";
import feedJson from "./fixtures/feed.json";
import graphJson from "./fixtures/graph.json";
import healthJson from "./fixtures/health.json";
import runJson from "./fixtures/run.json";
import runsJson from "./fixtures/runs.json";
import settingsJson from "./fixtures/settings.json";
import workflowJson from "./fixtures/workflow.json";

export const board = boardJson as Board;
export const boardEmpty = boardEmptyJson as Board;
export const changes = changesJson as Changes;
export const explain = explainJson as Explain;
export const feed = feedJson as FeedItem[];
export const graph = graphJson as Graph;
export const health = healthJson as Health;
export const run = runJson as RunDetail;
export const runs = runsJson as RunPage;
export const settings = settingsJson as Settings;
export const workflow = workflowJson as WorkflowDetail;

/** A `fetch` stub answering each API path (without its query) from `routes`; anything else is a 404. */
export function fetchStub(routes: Record<string, unknown>, status = 200) {
  return async (input: RequestInfo | URL): Promise<Response> => {
    const url = new URL(String(input), "http://localhost");
    if (!(url.pathname in routes)) {
      return new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 });
    }
    return new Response(JSON.stringify(routes[url.pathname]), {
      status,
      headers: { "content-type": "application/json" },
    });
  };
}
