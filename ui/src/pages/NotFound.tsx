import { Link, isRouteErrorResponse, useRouteError } from "react-router";
import { PATHS } from "../routes";

export function NotFound() {
  return (
    <section>
      <h1 className="page-title">Page not found</h1>
      <p>
        Nothing lives at this address. <Link to={PATHS.pitWall}>Back to the pit wall</Link>
      </p>
    </section>
  );
}

/** The router's error boundary: a render error shows a plain message, never a blank page. */
export function RouteError() {
  const error = useRouteError();
  const detail = isRouteErrorResponse(error)
    ? `${error.status} ${error.statusText}`
    : error instanceof Error
      ? error.message
      : String(error);
  return (
    <main className="page">
      <div className="error-state" role="alert">
        <p className="error-title">Something broke on this page</p>
        <p className="error-detail">{detail}</p>
        <a className="btn" href={PATHS.pitWall}>
          Back to the pit wall
        </a>
      </div>
    </main>
  );
}
