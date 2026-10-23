// The pit wall before any run arrives: it teaches the few-lines setup instead of saying "nothing here".
import { useEffect, useRef, useState } from "react";

const INSTALL = "uv sync";
const PUBLISHED = 'pip install "pitstop[ui]"';
const SNIPPET = [
  "import pitstop",
  "from openai import OpenAI",
  "",
  "client = pitstop.wrap(OpenAI())",
  'with pitstop.workflow("my-agent", criticality="medium"):',
  "    run_agent(client)",
].join("\n");
const SERVE = "pitstop ui";
const DEMO = "python examples/quickstart/dashboard_demo.py --home /tmp/pitstop-demo";
const DEMO_SERVE = "PITSTOP_HOME=/tmp/pitstop-demo pitstop ui";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      clearTimeout(timer.current);
      timer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      // no clipboard (insecure context, denied permission): the text stays selectable
    }
  }

  return (
    <button type="button" className="btn btn-ghost btn-copy" onClick={copy} aria-label={`Copy ${label}`}>
      <span aria-live="polite">{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function Code({ text, label, snippet }: { text: string; label: string; snippet?: string }) {
  return (
    <div className="code">
      <pre data-snippet={snippet}>
        <code>{text}</code>
      </pre>
      <CopyButton text={text} label={label} />
    </div>
  );
}

export function EmptyState() {
  return (
    <section className="empty" aria-labelledby="empty-title">
      <header className="empty-head">
        <h1 id="empty-title" className="page-title">
          No workflows yet
        </h1>
        <p className="lede">
          PitStop fills this wall as your agents run: one tile per workflow, flagged when it drifts, with what it
          costs. Three steps get you there.
        </p>
      </header>

      <ol className="steps">
        <li>
          <h2>Install</h2>
          <p>In a clone of the PitStop repo:</p>
          <Code text={INSTALL} label="the install command" />
          <p className="muted">
            <code>{PUBLISHED}</code> once it is published.
          </p>
        </li>
        <li>
          <h2>Wrap your client and name the workflow</h2>
          <p>Every call through the wrapped client is metered and checked for drift, grouped by workflow.</p>
          <Code text={SNIPPET} label="the snippet" snippet="wrap" />
        </li>
        <li>
          <h2>Run your agent, then open the dashboard</h2>
          <Code text={SERVE} label="the dashboard command" />
          <p className="muted">This page checks for new runs every few seconds; no reload needed.</p>
        </li>
      </ol>

      <aside className="empty-demo" aria-labelledby="empty-demo-title">
        <h2 id="empty-demo-title">Just looking?</h2>
        <p>Seed a throwaway home with made-up runs (no keys, no network) and point the dashboard at it:</p>
        <Code text={DEMO} label="the demo command" />
        <Code text={DEMO_SERVE} label="the demo dashboard command" />
      </aside>
    </section>
  );
}
