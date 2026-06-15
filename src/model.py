from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from PIL import Image

from .config import AppConfig
from .ocr import OCRLine, OCRResult
from .prompt import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

# API 重试配置
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 2
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class AnswerResult:
    answer: str
    evidence: list[str]
    confidence: str
    uncertainty: str
    raw_text: str
    provider: str
    model: str
    used_fallback: bool = False


class VisionQuestionAnswerer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.evidence_prompt_enabled: bool = True

    def answer(
        self,
        image: Image.Image,
        image_bytes: bytes,
        question: str,
        ocr_result: OCRResult,
        history: list[dict[str, str]],
    ) -> AnswerResult:
        if self.config.provider == "dashscope" and self.config.api_key:
            try:
                raw_text = self._call_dashscope(image_bytes, question, ocr_result, history, self.evidence_prompt_enabled)
                parsed = self._parse_model_output(raw_text)
                return AnswerResult(
                    answer=parsed["answer"],
                    evidence=parsed["evidence"],
                    confidence=parsed["confidence"],
                    uncertainty=parsed["uncertainty"],
                    raw_text=raw_text,
                    provider="dashscope",
                    model=self.config.model,
                )
            except Exception as exc:
                logger.warning("DashScope 调用失败，降级到 heuristic：%s", exc)
                fallback = self._heuristic_answer(question, ocr_result, str(exc))
                fallback.provider = "heuristic"
                fallback.model = self.config.model
                fallback.used_fallback = True
                return fallback

        if self.config.provider == "local":
            try:
                raw_text = self._call_local_model(image, image_bytes, question, ocr_result, history, self.evidence_prompt_enabled)
                parsed = self._parse_model_output(raw_text)
                return AnswerResult(
                    answer=parsed["answer"],
                    evidence=parsed["evidence"],
                    confidence=parsed["confidence"],
                    uncertainty=parsed["uncertainty"],
                    raw_text=raw_text,
                    provider="local",
                    model=self.config.model,
                )
            except Exception as exc:
                logger.warning("本地模型推理失败，降级到 heuristic：%s", exc)
                fallback = self._heuristic_answer(question, ocr_result, str(exc))
                fallback.provider = "heuristic"
                fallback.model = self.config.model
                fallback.used_fallback = True
                return fallback

        fallback = self._heuristic_answer(question, ocr_result, "provider disabled or key missing")
        fallback.provider = "heuristic"
        fallback.model = self.config.model
        fallback.used_fallback = True
        return fallback

    # ------------------------------------------------------------------
    # 本地模型推理
    # ------------------------------------------------------------------
    _local_model: Any = None
    _local_processor: Any = None

    def _get_local_model(self):
        if VisionQuestionAnswerer._local_model is not None:
            return VisionQuestionAnswerer._local_model, VisionQuestionAnswerer._local_processor

        import torch
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            model_cls = AutoModelForImageTextToText
        except Exception:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            model_cls = AutoModelForVision2Seq

        model_path = self.config.local_model_path
        device_map = self.config.local_device
        if device_map == "auto" and not torch.cuda.is_available():
            device_map = "cpu"

        logger.info("正在加载本地模型 %s (device=%s) ...", model_path, device_map)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = model_cls.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map if device_map == "auto" else None,
        )
        if device_map not in ("auto",) and not torch.cuda.is_available():
            model = model.to(device_map)
        if getattr(processor, "tokenizer", None) is not None and getattr(processor.tokenizer, "pad_token", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        # 加载 LoRA adapter（如果配置了 lora_path）
        if self.config.lora_path:
            logger.info("正在加载 LoRA adapter: %s", self.config.lora_path)
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.config.lora_path)
                model = model.merge_and_unload()
                logger.info("LoRA adapter 已合并。")
            except Exception as exc:
                logger.warning("LoRA adapter 加载失败，继续使用基座模型: %s", exc)

        model.eval()

        VisionQuestionAnswerer._local_model = model
        VisionQuestionAnswerer._local_processor = processor
        return model, processor

    def _call_local_model(
        self,
        image: Image.Image,
        image_bytes: bytes,
        question: str,
        ocr_result: OCRResult,
        history: list[dict[str, str]],
        evidence_enabled: bool = True,
    ) -> str:
        import torch
        model, processor = self._get_local_model()
        user_prompt = build_user_prompt(question, ocr_result, history, self.config.max_ocr_lines, evidence_enabled)

        messages: list[dict[str, Any]] = [{"role": "system", "content": [{"type": "text", "text": build_system_prompt(evidence_enabled)}]}]
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": [{"type": "text", "text": content.strip()}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ],
        })

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=False,
            )

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        raw_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        return raw_text.strip()

    def _call_dashscope(
        self,
        image_bytes: bytes,
        question: str,
        ocr_result: OCRResult,
        history: list[dict[str, str]],
        evidence_enabled: bool = True,
    ) -> str:
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{image_data}"
        user_prompt = build_user_prompt(question, ocr_result, history, self.config.max_ocr_lines, evidence_enabled)

        messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt(evidence_enabled)}]
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content.strip():
                messages.append({"role": role, "content": content.strip()})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        })

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        last_exception: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.config.base_url,
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.config.request_timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                if response.status_code in RETRYABLE_STATUSES and attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "DashScope API 返回 %d，第 %d/%d 次重试，等待 %.1fs ...",
                        response.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                response.raise_for_status()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exception = exc
                if attempt < MAX_RETRIES - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "DashScope API 网络错误：%s，第 %d/%d 次重试，等待 %.1fs ...",
                        exc,
                        attempt + 1,
                        MAX_RETRIES,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    raise
        raise last_exception or RuntimeError("DashScope API 调用在多次重试后仍失败")

    def _heuristic_answer(self, question: str, ocr_result: OCRResult, reason: str) -> AnswerResult:
        # 闲聊检测：打招呼等简短对话直接友好回应
        casual_set = {
            "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "在吗",
            "你是谁", "你叫什么", "介绍一下",
            "谢谢", "感谢", "再见", "拜拜", "晚安",
            "早上好", "下午好", "晚上好", "哈哈", "呵呵",
        }
        question_stripped = question.strip().lower().rstrip("?？！!.。")
        if question_stripped in casual_set or len(question_stripped) <= 3:
            return AnswerResult(
                answer=f"你好！我是中文图文问答助手，可以帮你分析上传的图片内容。有什么可以帮你的吗？",
                evidence=[],
                confidence="高",
                uncertainty="",
                raw_text=json.dumps({"answer": "你好！我是中文图文问答助手。", "evidence": [], "confidence": "高", "uncertainty": ""}, ensure_ascii=False),
                provider="heuristic",
                model=self.config.model,
                used_fallback=True,
            )

        selected_lines = self._select_key_lines(question, ocr_result)
        answer = self._summarize_question(question, selected_lines, ocr_result.raw_text, ocr_result.analysis)
        evidence = selected_lines[:3]
        if not evidence and ocr_result.raw_text:
            evidence = [ocr_result.raw_text[:180]]
        if not evidence:
            evidence = ["当前未获得可用OCR文本，无法从图像中稳定定位证据。"]
        uncertainty = reason
        if ocr_result.backend == "none":
            uncertainty = "未识别到可用OCR后端，当前为保守兜底回答。"
        elif ocr_result.analysis is not None:
            if ocr_result.analysis.likely_table:
                uncertainty = "当前图像疑似表格，兜底答案优先依据表格中的关键字段，不代表完整解析。"
            elif ocr_result.analysis.likely_chart:
                uncertainty = "当前图像疑似图表，兜底答案优先依据数字和趋势，不代表完整曲线解读。"
        confidence = "低" if ocr_result.backend == "none" else "中"
        return AnswerResult(
            answer=answer,
            evidence=evidence,
            confidence=confidence,
            uncertainty=uncertainty,
            raw_text=json.dumps(
                {
                    "answer": answer,
                    "evidence": evidence,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                },
                ensure_ascii=False,
            ),
            provider="heuristic",
            model=self.config.model,
            used_fallback=True,
        )

    @staticmethod
    def _select_key_lines(question: str, ocr_result: OCRResult) -> list[str]:
        lines = [line for line in ocr_result.lines if line.text.strip()]
        if not lines:
            return []

        question_lower = question.lower()
        noise_tokens = ("问题示例", "图例", "说明", "注：", "备注", "标题", "题目")
        chart_keywords = ("图表", "趋势", "增长", "下降", "最高", "最低", "季度", "变化", "上升", "下跌")

        def is_noise(text: str) -> bool:
            lowered = text.lower()
            return any(token.lower() in lowered for token in noise_tokens)

        def is_chart_signal(text: str) -> bool:
            compact = re.sub(r"[\s\W_]+", "", text)
            has_digits = any(char.isdigit() for char in text)
            has_chinese = any("\u4e00" <= char <= "\u9fff" for char in text)
            if not has_digits:
                return len(compact) <= 6
            if re.fullmatch(r"q\d+(?:[-–—]q\d+)?", compact.lower()):
                return True
            if re.fullmatch(r"[a-z]+\d+(?:[-–—][a-z]+\d+)?", compact.lower()):
                return True
            if len(text) <= 12 and not has_chinese:
                return True
            return len(text) <= 8 and not has_chinese

        def layout_key(line: OCRLine) -> tuple[int, int, int]:
            if line.bbox is None:
                return (1, 0, 0)
            x1, y1, _, _ = line.bbox
            return (0, x1, y1)

        def score(text: str) -> float:
            value = 0.0
            if any(char.isdigit() for char in text):
                value += 3.0
            if any(token in question_lower for token in chart_keywords):
                if len(text) <= 20:
                    value += 1.5
            if any(char.isalpha() for char in text) and len(text) <= 12:
                value += 1.0
            if len(text) >= 20:
                value -= 1.5
            if is_noise(text):
                value -= 4.0
            return value

        ranked = sorted(lines, key=lambda line: score(line.text), reverse=True)
        primary = [line.text for line in ranked if score(line.text) > 0][:8]
        if ocr_result.analysis is not None and ocr_result.analysis.likely_chart:
            chart_lines = [line for line in lines if is_chart_signal(line.text)]
            chart_lines = [line for line in chart_lines if not is_noise(line.text)]
            chart_lines = sorted(chart_lines, key=layout_key)
            if chart_lines:
                numeric_lines = [line.text for line in chart_lines if VisionQuestionAnswerer._extract_numeric_value(line.text) is not None]
                primary = numeric_lines[:8] if len(numeric_lines) >= 2 else [line.text for line in chart_lines[:8]]
        elif ocr_result.analysis is not None and ocr_result.analysis.likely_table:
            table_lines = [line.text for line in lines if len(line.text) <= 18 and not is_noise(line.text)]
            if table_lines:
                primary = table_lines[:8]

        return primary or lines[:6]

    @staticmethod
    def _summarize_question(question: str, selected_lines: list[str], raw_text: str, ocr_analysis=None) -> str:
        question_lower = question.lower()
        if not selected_lines and raw_text:
            selected_lines = [raw_text[:120]]

        # 基于问题关键词与 OCR 分析综合判定表格/图表上下文
        is_table_context = ocr_analysis and ocr_analysis.likely_table
        is_chart_context = ocr_analysis and ocr_analysis.likely_chart

        # 问题中明确提到"图表"等关键词时，优先按图表处理
        question_has_chart_signal = any(token in question_lower for token in ("图表", "柱状图", "折线图", "饼图", "散点", "走势", "季度", "月份", "年份", "x轴", "y轴"))
        question_has_table_signal = any(token in question_lower for token in ("表格", "列表", "清单", "第几行", "第几列", "单元格"))

        if question_has_chart_signal and not question_has_table_signal:
            is_table_context = False
            is_chart_context = True
        elif question_has_table_signal and not question_has_chart_signal:
            is_table_context = True
            is_chart_context = False

        if is_table_context and any(token in question_lower for token in ("最高", "最低", "最大", "最小", "最多", "最少", "准确率", "分数", "数值", "数据", "哪一", "哪个", "哪种", "是否", "有没有", "什么", "多少", "几列", "几行")):
            if not question_has_chart_signal:
                matched = next((line for line in selected_lines if any(ch in line for ch in question_lower[:4])), "")
                if not matched and selected_lines:
                    matched = selected_lines[0]
                if matched:
                    return f"根据表格数据，与问题相关的内容为：{matched}。"

        if not is_table_context and any(token in question_lower for token in ("趋势", "变化", "增长", "下降", "对比", "最高", "最低", "上升", "下跌")):
            chart_answer = VisionQuestionAnswerer._summarize_chart_signal(selected_lines)
            if chart_answer:
                return chart_answer

        if any(token in question_lower for token in ("标题", "题目", "名称", "名字")):
            return f"从OCR中看，较可能的标题或名称是：{selected_lines[0] if selected_lines else raw_text[:80]}"

        if any(token in question_lower for token in ("多少", "几", "数量", "人数", "金额", "价格", "分数", "比例")):
            numeric_snippet = next((line for line in selected_lines if any(char.isdigit() for char in line)), None)
            if numeric_snippet:
                return f"图中包含数字线索：{numeric_snippet}。若你要的是具体数值，建议结合原图进一步核对。"

        if selected_lines:
            return f"根据当前证据，图中可读到的关键信息包括：{'；'.join(selected_lines[:3])}。"

        return "当前没有足够清晰的OCR证据，我只能给出保守判断：无法从现有输入中稳定得出答案。"

    @staticmethod
    def _summarize_chart_signal(selected_lines: list[str]) -> str:
        if not selected_lines:
            return "当前没有足够的数字或文字线索，无法稳定判断图表趋势。"

        series: list[tuple[str, float | None]] = []
        for line in selected_lines:
            value = VisionQuestionAnswerer._extract_numeric_value(line)
            if value is None:
                continue
            label = re.sub(r"(?<![A-Za-z])(?:-?\d+(?:\.\d+)?)(?:%|万|千|百|亿)?(?![A-Za-z])", "", line).strip(" ：:，,|\t")
            series.append((label or line, value))

        values = [value for _, value in series if value is not None]
        if len(values) >= 2:
            first_value = values[0]
            last_value = values[-1]
            max_value = max(values)
            min_value = min(values)
            max_index = values.index(max_value)
            min_index = values.index(min_value)

            trend = "整体上升" if last_value > first_value else "整体下降" if last_value < first_value else "整体持平"
            max_text = series[max_index][0]
            min_text = series[min_index][0]

            details = [f"根据OCR中的数值线索，图表走势{trend}。", f"数值范围大致在 {min_value:g} 到 {max_value:g} 之间。"]
            if max_text and max_text != str(max_value):
                details.append(f"最高值对应的线索是“{max_text}”。")
            if min_text and min_text != str(min_value):
                details.append(f"最低值对应的线索是“{min_text}”。")
            if any(token in "".join(selected_lines) for token in ("%", "百分", "占比", "比例")):
                details.append("这类图表更适合结合百分比或占比进行比较。")
            return "".join(details)

        if series:
            labels = "；".join(label for label, _ in series[:3])
            return f"当前图表包含这些可读数值线索：{labels}。由于数值不够完整，我只能给出保守判断：请结合原图确认趋势。"

        if len(selected_lines) >= 2:
            return f"图中可见的关键文本包括：{'；'.join(selected_lines[:3])}。看起来更像图表或带数值的版面，但数字线索不足，无法稳定判断趋势。"

        return "当前没有足够的数字或文字线索，无法稳定判断图表趋势。"

    @staticmethod
    def _extract_numeric_value(text: str) -> float | None:
        compact = text.strip().replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?%?", compact):
            return float(compact.rstrip("%"))
        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:万|千|百|亿)?", compact):
            return float(re.sub(r"[万千百亿]", "", compact))
        return None

    @staticmethod
    def _parse_model_output(raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()
        try:
            parsed = json.loads(raw_text)
            return {
                "answer": str(parsed.get("answer", raw_text)).strip(),
                "evidence": [str(item) for item in parsed.get("evidence", []) if str(item).strip()],
                "confidence": str(parsed.get("confidence", "中")).strip() or "中",
                "uncertainty": str(parsed.get("uncertainty", "")).strip(),
            }
        except Exception:
            pass

        answer = raw_text
        evidence: list[str] = []
        confidence = "中"
        uncertainty = ""
        for line in raw_text.splitlines():
            normalized = line.strip()
            if normalized.startswith(("答案：", "答：")):
                answer = normalized.split("：", 1)[-1].strip()
            elif normalized.startswith(("证据：", "依据：")):
                evidence_text = normalized.split("：", 1)[-1].strip()
                evidence = [part.strip() for part in evidence_text.split("；") if part.strip()]
            elif normalized.startswith("置信度："):
                confidence = normalized.split("：", 1)[-1].strip() or "中"
            elif normalized.startswith(("不确定：", "无法判断：")):
                uncertainty = normalized.split("：", 1)[-1].strip()

        return {
            "answer": answer,
            "evidence": evidence,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }
