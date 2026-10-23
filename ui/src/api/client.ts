// The dashboard's typed API client. Types come from `schema.d.ts`, generated from the server's OpenAPI schema
// (`npm run gen:api`; `tests/server/test_openapi_snapshot.py` keeps `openapi.json` current).
//
// Workflow names travel as query parameters, never path segments (decision 5), built with URLSearchParams so a name
// holding "/", "?", "#", spaces or emoji round-trips. Every failure throws an ApiError with a message fit to show.
import type { components } from "./schema";

type Schemas = components["schemas"];
export type Board = Schemas["Board"];
export type BudgetRow = Schemas["BudgetRow"];
export type Changes = Schemas["Changes"];
export type DismissResult = Schemas["DismissResult"];
export type DriftCard = Schemas["DriftCard"];
export type EventRow = Schemas["EventRow"];
export type Explain = Schemas["Explain"];
export type FeedItem = Schemas["FeedItem"];
export type Graph = Schemas["Graph"];
export type GraphNode = Schemas["GraphNode"];
export type Health = Schemas["Health"];
export type Marker = Schemas["Marker"];
export type MetricInfo = Schemas["MetricInfo"];
export type RunDetail = Schemas["RunDetail"];
export type RunPage = Schemas["RunPage"];
export type RunRow = Schemas["RunRow"];
export type Settings = Schemas["Settings"];
export type Tile = Schemas["Tile"];
export type WorkerStatus = Schemas["WorkerStatus"];
export type WorkflowDetail = Schemas["WorkflowDetail"];
export type GraphWindow = "1h" | "24h" | "7d" | "30d";

/** A failed API call. `status` is the HTTP status, or 0 when the server could not be reached. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Per-call options: `signal` aborts the request (usePoll passes one). */
export interface CallOptions {
  signal?: AbortSignal;
}

type Query = Record<string, string | number | null | undefined>;

function url(path: string, query: Query = {}): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value !== null && value !== undefined) params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** The message in an error body: `{"error": str}` (PitStop's guards), `{"detail": str}` (an HTTPException) or
 *  `{"detail": [{loc, msg}, ...]}` (FastAPI's request validation). */
export function errorMessage(status: number, body: unknown): string {
  if (isRecord(body)) {
    if (typeof body.error === "string" && body.error) return body.error;
    const detail = body.detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) => {
          if (!isRecord(item)) return String(item);
          const loc = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "query" && p !== "path" && p !== "body") : [];
          const msg = typeof item.msg === "string" ? item.msg : "invalid";
          return loc.length ? `${loc.join(".")}: ${msg}` : msg;
        })
        .join("; ");
    }
  }
  return `The server answered ${status}.`;
}

async function request<T>(method: "GET" | "POST", path: string, opts: CallOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") headers["X-PitStop"] = "1"; // the server's write guard (decision 2)
  let response: Response;
  try {
    response = await fetch(path, { method, headers, signal: opts.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError(0, "Can't reach the PitStop server.");
  }
  let body: unknown = null;
  try {
    body = await response.json();
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    body = null; // not JSON, e.g. a proxy's error page
  }
  if (!response.ok) throw new ApiError(response.status, errorMessage(response.status, body));
  return body as T;
}

const get = <T>(path: string, query?: Query, opts?: CallOptions) => request<T>("GET", url(path, query), opts);

export const getHealth = (opts?: CallOptions) => get<Health>("/api/health", {}, opts);

export const getBoard = (opts?: CallOptions) => get<Board>("/api/board", {}, opts);

export const getFeed = (limit: number, opts?: CallOptions) => get<FeedItem[]>("/api/feed", { limit }, opts);

export const getWorkflow = (name: string, opts?: CallOptions) => get<WorkflowDetail>("/api/workflow", { name }, opts);

/** Dismiss an open drift. A drift already gone (404) or no longer open (409) throws with a plain message. */
export async function dismissDrift(id: number, opts?: CallOptions): Promise<DismissResult> {
  try {
    return await request<DismissResult>("POST", `/api/drifts/${id}/dismiss`, opts);
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) throw new ApiError(404, "This drift no longer exists.");
    if (e instanceof ApiError && e.status === 409) throw new ApiError(409, "This drift is no longer open.");
    throw e;
  }
}

export const explainDrift = (id: number, opts?: CallOptions) => get<Explain>(`/api/drifts/${id}/explain`, {}, opts);

/** One page of runs, newest first; pass the previous page's `next_before` as `before` for the next. */
export const getRuns = (workflow: string, before?: number | null, opts?: CallOptions) =>
  get<RunPage>("/api/runs", { workflow, before }, opts);

export const getRun = (id: string, opts?: CallOptions) => get<RunDetail>(`/api/runs/${encodeURIComponent(id)}`, {}, opts);

/** The run graph for one run (`runId`) or the newest runs started in `window`. */
export const getGraph = (workflow: string, scope: { runId?: string | null; window: GraphWindow }, opts?: CallOptions) =>
  get<Graph>("/api/graph", { workflow, run_id: scope.runId || null, window: scope.window }, opts);

export const getChanges = (workflow: string, opts?: CallOptions) => get<Changes>("/api/changes", { workflow }, opts);

export const getSettings = (opts?: CallOptions) => get<Settings>("/api/settings", {}, opts);
