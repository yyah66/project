from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any


@dataclass
class AppConfig:
    provider: str = field(
        default_factory=lambda: os.getenv("VLM_PROVIDER", "dashscope").strip().lower()
    )
    model: str = field(
        default_factory=lambda: os.getenv("VLM_MODEL", "qwen3-vl-flash").strip()
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "VLM_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ).strip()
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    )
    local_model_path: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
    )
    local_device: str = field(
        default_factory=lambda: os.getenv("LOCAL_DEVICE", "auto").strip()
    )
    lora_path: str = field(
        default_factory=lambda: os.getenv("LORA_PATH", "").strip()
    )
    max_ocr_lines: int = field(
        default_factory=lambda: int(os.getenv("MAX_OCR_LINES", "24"))
    )
    max_history_turns: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_TURNS", "6"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60"))
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "max_ocr_lines": self.max_ocr_lines,
            "max_history_turns": self.max_history_turns,
            "request_timeout": self.request_timeout,
        }


def load_config() -> AppConfig:
    return AppConfig()
