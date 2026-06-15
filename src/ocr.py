from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass
class OCRLine:
    text: str
    bbox: tuple[int, int, int, int] | None = None
    confidence: float | None = None


@dataclass
class OCRResult:
    backend: str
    lines: list[OCRLine]
    analysis: "ImageAnalysis | None" = None

    @property
    def raw_text(self) -> str:
        return "\n".join(line.text for line in self.lines if line.text.strip())


@dataclass
class ImageAnalysis:
    scene_type: str
    text_density: float
    likely_table: bool
    likely_chart: bool
    likely_document: bool
    language_hint: str
    notes: list[str]


class NoOpOCRService:
    """空 OCR 服务：用于消融实验关闭 OCR 时的占位实现。"""

    def extract(self, image: Image.Image, fast_mode: bool = True) -> OCRResult:
        analysis = ImageAnalysis(
            scene_type="未启用OCR",
            text_density=0.0,
            likely_table=False,
            likely_chart=False,
            likely_document=False,
            language_hint="",
            notes=["OCR 已关闭，仅使用视觉信息。"],
        )
        return OCRResult(backend="ocr_disabled", lines=[], analysis=analysis)


class OCRService:
    def __init__(self, providers: list[str] | None = None) -> None:
        self._paddleocr = None
        self._rapidocr = None
        self._pytesseract = None
        self._paddleocr_lang = os.getenv("PADDLEOCR_LANG", "ch").strip() or "ch"
        self._rapidocr_providers = providers

    def extract(self, image: Image.Image, fast_mode: bool = True) -> OCRResult:
        # ---- 快速模式：仅对原图运行 rapidocr，命中则直接返回 ----
        if fast_mode:
            try:
                lines = self._extract_with_rapidocr(image)
                lines = self._normalize_lines(lines)
                if lines:
                    analysis = self.analyze_image(image, lines)
                    return OCRResult(backend="rapidocr-onnxruntime", lines=lines, analysis=analysis)
            except Exception:
                pass
            # 快速模式 rapidocr 无结果时退到 paddle 单原图
            try:
                lines = self._extract_with_paddleocr(image)
                lines = self._normalize_lines(lines)
                if lines:
                    analysis = self.analyze_image(image, lines)
                    return OCRResult(backend="paddleocr", lines=lines, analysis=analysis)
            except Exception:
                pass

        # ---- 完整模式：多后端 + 图像变体增强 ----
        for backend_name, extractor in (
            ("rapidocr-onnxruntime", self._extract_with_rapidocr),
            ("paddleocr", self._extract_with_paddleocr),
            ("pytesseract", self._extract_with_pytesseract),
        ):
            try:
                lines = self._extract_best_lines(extractor, image)
                if lines:
                    analysis = self.analyze_image(image, lines)
                    return OCRResult(backend=backend_name, lines=lines, analysis=analysis)
            except Exception:
                continue
        return OCRResult(backend="none", lines=[], analysis=self.analyze_image(image, []))

    def _extract_best_lines(self, extractor: Any, image: Image.Image) -> list[OCRLine]:
        candidates: list[tuple[float, list[OCRLine], str]] = []
        for variant_name, variant_image in self._build_variants(image):
            try:
                lines = extractor(variant_image)
            except Exception:
                continue
            normalized = self._normalize_lines(lines)
            if not normalized:
                continue
            score = self._score_lines(normalized)
            if variant_name != "原图":
                score += 0.05
            candidates.append((score, normalized, variant_name))

        if not candidates:
            return []

        best_score, best_lines, _ = max(candidates, key=lambda item: item[0])
        if best_score <= 0:
            return []

        merged = self._merge_lines([lines for _, lines, _ in sorted(candidates, key=lambda item: item[0], reverse=True)[:2]])
        return merged or best_lines

    @staticmethod
    def _build_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
        base = image.convert("RGB")
        gray = ImageOps.grayscale(base)
        contrast = ImageEnhance.Contrast(gray).enhance(1.8)
        sharpened = contrast.filter(ImageFilter.SHARPEN).filter(ImageFilter.EDGE_ENHANCE_MORE)
        autocontrast = ImageOps.autocontrast(gray)
        upscaled = autocontrast.resize((max(1, base.width * 2), max(1, base.height * 2)), Image.Resampling.LANCZOS)
        binary = autocontrast.point(lambda pixel: 255 if pixel > 180 else 0)
        top_crop = OCRService._crop_and_resize(base, 0.0, 0.42)
        middle_crop = OCRService._crop_and_resize(base, 0.18, 0.82)
        bottom_crop = OCRService._crop_and_resize(base, 0.58, 1.0)
        return [
            ("原图", base),
            ("灰度增强", autocontrast),
            ("锐化增强", sharpened),
            ("二值化", binary),
            ("放大增强", upscaled),
            ("顶部裁剪", top_crop),
            ("中部裁剪", middle_crop),
            ("底部裁剪", bottom_crop),
        ]

    @staticmethod
    def _crop_and_resize(image: Image.Image, top_ratio: float, bottom_ratio: float) -> Image.Image:
        width, height = image.size
        top = max(0, min(height - 1, int(height * top_ratio)))
        bottom = max(top + 1, min(height, int(height * bottom_ratio)))
        crop = image.crop((0, top, width, bottom)).convert("RGB")
        long_side = max(crop.size)
        if long_side < 1200:
            scale = 1200 / float(max(long_side, 1))
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
        return crop

    @staticmethod
    def _normalize_lines(lines: list[OCRLine]) -> list[OCRLine]:
        normalized: list[OCRLine] = []
        seen: set[str] = set()
        for line in lines:
            text = OCRService._clean_text(line.text)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(OCRLine(text=text, bbox=line.bbox, confidence=line.confidence))
        return normalized

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = str(text).strip()
        cleaned = cleaned.replace("\u3000", " ")
        cleaned = cleaned.replace("�", "")
        cleaned = " ".join(cleaned.split())
        return cleaned

    @staticmethod
    def _score_lines(lines: list[OCRLine]) -> float:
        score = 0.0
        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            score += min(len(text), 24)
            if any(char.isdigit() for char in text):
                score += 2.0
            if any(_contains_cjk(char) for char in text):
                score += 2.0
            if line.confidence is not None:
                score += max(0.0, line.confidence)
        if len(lines) >= 5:
            score += 1.5
        return score

    @staticmethod
    def _merge_lines(candidate_groups: list[list[OCRLine]]) -> list[OCRLine]:
        merged: list[OCRLine] = []
        seen: set[str] = set()
        for group in candidate_groups:
            for line in group:
                key = OCRService._clean_text(line.text).lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(line)
        return merged

    @staticmethod
    def analyze_image(image: Image.Image, lines: list[OCRLine]) -> ImageAnalysis:
        width, height = image.size
        area = max(width * height, 1)
        text_chars = sum(len(line.text.strip()) for line in lines)
        text_density = min(text_chars / area * 8000.0, 1.0)

        digit_lines = sum(1 for line in lines if any(char.isdigit() for char in line.text))
        long_lines = sum(1 for line in lines if len(line.text.strip()) >= 12)
        short_lines = sum(1 for line in lines if 1 <= len(line.text.strip()) <= 4)
        separator_lines = sum(1 for line in lines if any(sep in line.text for sep in ("|", "\t", "-")))

        likely_table = len(lines) >= 6 and (separator_lines >= 2 or short_lines >= 3)
        likely_chart = len(lines) >= 3 and digit_lines >= 2 and long_lines <= max(1, len(lines) // 2)

        # 纯数值图：短文本几乎全是数字且缺少中文标签 → 优先判定为图表
        numeric_short_count = sum(1 for line in lines if 1 <= len(line.text.strip()) <= 4 and line.text.strip().replace(".", "").replace("-", "").isdigit())
        text_label_count = sum(1 for line in lines if len(line.text.strip()) >= 2 and not line.text.strip().replace(".", "").replace("-", "").replace(",", "").replace("%", "").isdigit())
        if likely_table and numeric_short_count >= 3 and text_label_count <= 2:
            likely_table = False

        likely_document = len(lines) >= 5 and long_lines >= 3 and not likely_table

        if likely_table:
            scene_type = "表格/清单"
        elif likely_chart:
            scene_type = "图表/数值图"
        elif likely_document:
            scene_type = "文档/课件"
        elif len(lines) <= 2:
            scene_type = "自然场景/单句文本"
        else:
            scene_type = "混合版面"

        notes: list[str] = []
        if likely_table:
            notes.append("版面中存在较多短文本或分隔符，疑似表格结构。")
        if likely_chart:
            notes.append("检测到较多数字线索，疑似图表或统计信息。")
        if likely_document:
            notes.append("检测到较多长文本行，疑似文档或课件。")
        if text_density < 0.005:
            notes.append("OCR文本密度偏低，图片可能以自然场景为主。")

        language_hint = "中文/混合文本" if any(_contains_cjk(line.text) for line in lines) else "未检测到明显中文"

        return ImageAnalysis(
            scene_type=scene_type,
            text_density=round(text_density, 4),
            likely_table=likely_table,
            likely_chart=likely_chart,
            likely_document=likely_document,
            language_hint=language_hint,
            notes=notes,
        )

    def _extract_with_paddleocr(self, image: Image.Image) -> list[OCRLine]:
        try:
            if self._paddleocr is None:
                from paddleocr import PaddleOCR

                # use_textline_orientation 在 paddleocr >= 2.9 中已移除，先尝试带参数，失败则回退
                try:
                    self._paddleocr = PaddleOCR(use_textline_orientation=True, lang=self._paddleocr_lang)
                except TypeError:
                    self._paddleocr = PaddleOCR(lang=self._paddleocr_lang)

            result = self._paddleocr.ocr(np.array(image.convert("RGB")), cls=True)
            return self._parse_paddleocr_result(result)
        except Exception:
            return []

    def _extract_with_rapidocr(self, image: Image.Image) -> list[OCRLine]:
        try:
            if self._rapidocr is None:
                from rapidocr_onnxruntime import RapidOCR

                if self._rapidocr_providers:
                    self._rapidocr = RapidOCR(
                        det_providers=self._rapidocr_providers,
                        rec_providers=self._rapidocr_providers,
                        cls_providers=self._rapidocr_providers,
                    )
                else:
                    self._rapidocr = RapidOCR()
            result, _ = self._rapidocr(np.array(image.convert("RGB")))
            lines: list[OCRLine] = []
            if not result:
                return lines
            for item in result:
                if len(item) < 3:
                    continue
                box, text, confidence = item[0], str(item[1]).strip(), float(item[2])
                if not text:
                    continue
                bbox = self._bbox_from_polygon(box)
                lines.append(OCRLine(text=text, bbox=bbox, confidence=confidence))
            return lines
        except Exception:
            return []

    @staticmethod
    def _parse_paddleocr_result(result: Any) -> list[OCRLine]:
        lines: list[OCRLine] = []
        if not result:
            return lines

        for item in result:
            if not item:
                continue

            candidate = item
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], list):
                candidate = item[0]

            if not isinstance(candidate, list) or len(candidate) < 2:
                continue

            box = candidate[0]
            text_info = candidate[1]

            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text = str(text_info[0]).strip()
                confidence = None
                try:
                    confidence = float(text_info[1])
                except Exception:
                    confidence = None
            else:
                text = str(text_info).strip()
                confidence = None

            if not text:
                continue

            bbox = OCRService._bbox_from_polygon(box)
            lines.append(OCRLine(text=text, bbox=bbox, confidence=confidence))

        return lines

    def _extract_with_pytesseract(self, image: Image.Image) -> list[OCRLine]:
        try:
            if self._pytesseract is None:
                import pytesseract

                self._pytesseract = pytesseract
            data = self._pytesseract.image_to_data(image, output_type=self._pytesseract.Output.DICT)
            lines: list[OCRLine] = []
            total = len(data.get("text", []))
            for index in range(total):
                text = str(data["text"][index]).strip()
                if not text:
                    continue
                left = int(data["left"][index])
                top = int(data["top"][index])
                width = int(data["width"][index])
                height = int(data["height"][index])
                confidence_raw = data.get("conf", [None] * total)[index]
                confidence = None
                try:
                    confidence = float(confidence_raw) / 100.0 if confidence_raw not in (None, "-1") else None
                except Exception:
                    confidence = None
                lines.append(
                    OCRLine(
                        text=text,
                        bbox=(left, top, left + width, top + height),
                        confidence=confidence,
                    )
                )
            return lines
        except Exception:
            return []

    @staticmethod
    def _bbox_from_polygon(box: Any) -> tuple[int, int, int, int] | None:
        try:
            points = np.asarray(box)
            if points.size == 0:
                return None
            xs = points[:, 0]
            ys = points[:, 1]
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        except Exception:
            return None


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(file_bytes)).convert("RGB")


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
