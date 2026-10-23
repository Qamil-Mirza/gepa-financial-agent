// Run detail at /run?id=…; Task 14 builds it.
import { useSearchParams } from "react-router";
import { MissingParam } from "./Workflow";

export function RunDetail() {
  const [params] = useSearchParams();
  const id = params.get("id");
  if (!id) return <MissingParam what="run" />;
  return (
    <section>
      <h1 className="page-title">Run {id}</h1>
    </section>
  );
}
