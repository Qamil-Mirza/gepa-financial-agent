"""The committed OpenAPI schema (`ui/openapi.json`, from which `ui/src/api/schema.d.ts` is generated) matches the
server's models, so the dashboard's typed client cannot silently drift from the API."""
import json
from pathlib import Path

from pitstop.server.app import export_openapi

COMMITTED = Path(__file__).resolve().parents[2] / "ui" / "openapi.json"
REGEN = ("uv run python -c \"import json; from pitstop.server.app import export_openapi; "
         "open('ui/openapi.json', 'w').write(json.dumps(export_openapi(), indent=2, sort_keys=True) + '\\n')\"")


def test_committed_openapi_is_current():
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert committed == export_openapi(), (
        f"ui/openapi.json is out of date with the server's models. From the repo root run:\n  {REGEN}\n"
        "  npm --prefix ui run gen:api\nand commit both ui/openapi.json and ui/src/api/schema.d.ts.")
