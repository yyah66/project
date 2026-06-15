from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageOps

from .config import AppConfig
from .model import AnswerResult, VisionQuestionAnswerer
from .ocr import OCRResult, OCRService, load_image_from_bytes


@dataclass
class PreprocessResult:
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    rotation_applied: int
    auto_rotated: bool
    resized: bool
    steps: list[str]


@dataclass
class ConversationTurn:
    role: str
    content: str


class AssistantEngine:
    max_side: int = 1600

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.ocr_service = OCRService()
        self.answerer = VisionQuestionAnswerer(self.config)

    def _preprocess(self, image: Image.Image) -> tuple[Image.Image, PreprocessResult]:
        steps: list[str] = []
        original_size = image.size
        processed = ImageOps.exif_transpose(image).convert("RGB")
        if processed.size != original_size:
            steps.append("EXIF 方向修正")

        resized = False
        if max(processed.size) > self.max_side:
            w, h = processed.size
            scale = self.max_side / float(max(w, h))
            processed = processed.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
            resized = True
            steps.append(f"尺寸归一(长边={self.max_side})")

        if not steps:
            steps.append("无需预处理")

        result = PreprocessResult(
            original_size=original_size,
            processed_size=processed.size,
            rotation_applied=0,
            auto_rotated=False,
            resized=resized,
            steps=steps,
        )
        return processed, result

    def inspect_image(self, image_bytes: bytes, fast_ocr: bool = True) -> tuple[Image.Image, PreprocessResult, OCRResult]:
        image = load_image_from_bytes(image_bytes)
        processed_image, preprocess_result = self._preprocess(image)
        ocr_result = self.ocr_service.extract(processed_image, fast_mode=fast_ocr)
        return processed_image, preprocess_result, ocr_result

    @property
    def evidence_prompt_enabled(self) -> bool:
        return self.answerer.evidence_prompt_enabled

    @evidence_prompt_enabled.setter
    def evidence_prompt_enabled(self, value: bool) -> None:
        self.answerer.evidence_prompt_enabled = value

    def answer_question(
        self,
        image: Image.Image,
        image_bytes: bytes,
        question: str,
        history: list[ConversationTurn],
        ocr_result: OCRResult,
        *,
        full_ocr: bool = True,
    ) -> AnswerResult:
        # 提问时可选重新运行完整 OCR（多后端 + 变体增强）
        if full_ocr and ocr_result.backend in ("none", "ocr_disabled"):
            # 之前快速模式无结果，尝试完整模式
            ocr_result = self.ocr_service.extract(image, fast_mode=False)
        history_payload = [{"role": turn.role, "content": turn.content} for turn in history[-self.config.max_history_turns * 2 :]]
        return self.answerer.answer(image, image_bytes, question, ocr_result, history_payload)
