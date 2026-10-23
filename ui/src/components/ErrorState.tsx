// A failed load: what went wrong in plain words, and a way to try again.
import { ApiError } from "../api/client";

function explain(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) return "Can't reach the PitStop server. Is pitstop ui still running?";
    if (error.status === 503) return "The store is busy with another write. Trying again usually works.";
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

export interface ErrorStateProps {
  error: unknown;
  title?: string;
  onRetry?: () => void;
}

export function ErrorState({ error, title = "This didn't load", onRetry }: ErrorStateProps) {
  return (
    <div className="error-state" role="alert">
      <p className="error-title">{title}</p>
      <p className="error-detail">{explain(error)}</p>
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
