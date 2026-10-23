// The workflow page at /w?name=…&tab=…; Task 13 builds it.
import { Link, useSearchParams } from "react-router";
import { PATHS, workflowTab } from "../routes";

export function Workflow() {
  const [params] = useSearchParams();
  const name = params.get("name");
  if (!name) return <MissingParam what="workflow" />;
  return (
    <section>
      <h1 className="page-title">{name}</h1>
      <p className="muted">Tab: {workflowTab(params.get("tab"))}</p>
    </section>
  );
}

export function MissingParam({ what }: { what: string }) {
  return (
    <section>
      <h1 className="page-title">No {what} given</h1>
      <p>
        This link is missing its {what}. <Link to={PATHS.pitWall}>Back to the pit wall</Link>
      </p>
    </section>
  );
}
