// The dashboard's URLs. Workflow names and run ids travel as query parameters (decision 5), encoded with
// URLSearchParams, so names holding "/", "?", "#", spaces or emoji round-trip (Review Focus 3).

export const PATHS = {
  pitWall: "/",
  workflow: "/w",
  run: "/run",
  settings: "/settings",
} as const;

export const WORKFLOW_TABS = ["graph", "changes", "runs"] as const;
export type WorkflowTab = (typeof WORKFLOW_TABS)[number];

/** The workflow page: `/w?name=<name>&tab=<tab>` (no `tab` means the default, the run graph). */
export function workflowHref(name: string, tab?: WorkflowTab): string {
  const params = new URLSearchParams({ name });
  if (tab) params.set("tab", tab);
  return `${PATHS.workflow}?${params}`;
}

/** Run detail: `/run?id=<id>`. */
export function runHref(id: string): string {
  return `${PATHS.run}?${new URLSearchParams({ id })}`;
}

/** The `tab` query value as a known tab; anything else is the default, "graph". */
export function workflowTab(value: string | null): WorkflowTab {
  return (WORKFLOW_TABS as readonly string[]).includes(value ?? "") ? (value as WorkflowTab) : "graph";
}
