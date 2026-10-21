from __future__ import annotations

import os
from dataclasses import dataclass, field

CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1"
DEFAULT_SUPERVISOR_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
DEFAULT_REFLECTION_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    crusoe_api_key: str | None = field(default=None, repr=False)
    base_url: str = CRUSOE_BASE_URL
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    reflection_model: str = DEFAULT_REFLECTION_MODEL
    max_concurrent_inferences: int = 4
    thinking: bool | None = None
    json_mode: bool = False

    @property
    def has_key(self) -> bool:
        return bool(self.crusoe_api_key)

    @classmethod
    def from_env(cls, load_dotenv_file: bool = True) -> "Settings":
        if load_dotenv_file:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
        return cls(
            crusoe_api_key=os.environ.get("CRUSOE_API_KEY") or None,
            base_url=os.environ.get("CRUSOE_BASE_URL", CRUSOE_BASE_URL),
            supervisor_model=os.environ.get("SUPERVISOR_MODEL", DEFAULT_SUPERVISOR_MODEL),
            reflection_model=os.environ.get("REFLECTION_MODEL", DEFAULT_REFLECTION_MODEL),
            max_concurrent_inferences=int(os.environ.get("SUPERVISOR_MAX_CONCURRENCY", "4")),
            thinking=_parse_bool(os.environ.get("SUPERVISOR_THINKING")),
            json_mode=bool(_parse_bool(os.environ.get("SUPERVISOR_JSON_MODE"))),
        )
