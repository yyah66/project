from __future__ import annotations

from typing import Iterable

from .ocr import ImageAnalysis, OCRResult


def build_history_block(history: Iterable[dict[str, str]]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
    return "\n".join(lines)


def build_ocr_block(ocr_result: OCRResult, max_lines: int) -> str:
    if not ocr_result.lines:
        return "OCR结果：未提取到可用文本。"

    lines = ["OCR结果："]
    for index, line in enumerate(ocr_result.lines[:max_lines], start=1):
        bbox_text = ""
        if line.bbox:
            bbox_text = f" | bbox={line.bbox[0]},{line.bbox[1]},{line.bbox[2]},{line.bbox[3]}"
        confidence_text = f" | score={line.confidence:.3f}" if line.confidence is not None else ""
        lines.append(f"{index}. {line.text}{bbox_text}{confidence_text}")
    if len(ocr_result.lines) > max_lines:
        lines.append(f"... 还有 {len(ocr_result.lines) - max_lines} 行未显示")
    return "\n".join(lines)


def build_analysis_block(analysis: ImageAnalysis | None) -> str:
    if analysis is None:
        return "版式分析：未生成。"

    flags = []
    if analysis.likely_table:
        flags.append("表格")
    if analysis.likely_chart:
        flags.append("图表")
    if analysis.likely_document:
        flags.append("文档")
    flag_text = "、".join(flags) if flags else "无明显结构"
    note_text = "；".join(analysis.notes) if analysis.notes else "无额外提示"
    return (
        "版式分析：\n"
        f"- 场景类型：{analysis.scene_type}\n"
        f"- 文本密度：{analysis.text_density}\n"
        f"- 结构特征：{flag_text}\n"
        f"- 语言提示：{analysis.language_hint}\n"
        f"- 备注：{note_text}"
    )


def build_system_prompt(evidence_enabled: bool = True) -> str:
    json_example = '{"answer":"回答内容","evidence":["证据片段1","证据片段2"],"confidence":"高","uncertainty":""}'
    json_rules = (
        "【输出格式】"
        f"你必须只输出一行合法 JSON 字符串，不要输出任何其他文字。JSON 格式固定为：{json_example}\n"
        "字段说明：\n"
        "- answer: 最终回答（字符串）\n"
        "- evidence: 引用到的 OCR 片段或图像局部描述，闲聊时可设为空数组 []\n"
        "- confidence: 必须是三个选项之一：高、中、低\n"
        "- uncertainty: 如有不确定之处写在这里，否则为空字符串 \"\"\n"
        "重要：输出中不要包含任何 JSON 之外的内容，不要用 ```json 包裹，不要有任何解释或前缀。"
    )
    if evidence_enabled:
        return (
            "你是一个中文图文问答助手。用户上传了一张图片，你的首要任务是围绕图片内容回答问题。\n"
            "规则：\n"
            "1. 如果用户的问题是闲聊、打招呼或与图片无关的对话，请正常友好回应，不需要强行关联图片。\n"
            "2. 如果用户的问题与图片内容相关，必须优先依据图像和OCR证据回答。\n"
            "3. 如果证据不足，直接说明无法判断，不要编造。\n"
            "4. 如果包含版式分析，请优先判断是否是文档、表格或图表，并据此调整回答方式。\n"
            "回答时保持自然、简洁。\n"
            + json_rules
        )
    return (
        "你是一个中文图文问答助手。用户可以上传图片并向你提问。\n"
        "如果问题与图片无关（如打招呼、闲聊），正常友好回应即可。\n"
        "如果问题与图片相关，请根据图像内容回答。\n"
        "回答时保持自然、简洁，直接给出答案，不要输出 JSON 或其他格式。\n"
    )


def build_user_prompt(question: str, ocr_result: OCRResult, history: Iterable[dict[str, str]], max_ocr_lines: int, evidence_enabled: bool = True) -> str:
    history_block = build_history_block(history)
    ocr_block = build_ocr_block(ocr_result, max_ocr_lines)
    analysis_block = build_analysis_block(ocr_result.analysis)

    # 提前判断问题是否为闲聊
    casual_greetings = {"你好", "您好", "嗨", "哈喽", "hello", "hi", "hey", "在吗", "你是谁", "你叫什么", "介绍一下"}
    casual_phrases = {"谢谢", "感谢", "再见", "拜拜", "晚安", "早上好", "下午好", "晚上好", "哈哈", "呵呵"}
    question_stripped = question.strip().lower().rstrip("?？！!.。")
    is_casual = (
        question_stripped in casual_greetings
        or question_stripped in casual_phrases
        or len(question_stripped) <= 3
    )

    if is_casual:
        # 闲聊直接透传，不附加约束
        sections = [
            f"当前问题：{question}",
            "",
            "注意：这是闲聊或打招呼，请自然友好回应，无需关联图片内容。",
        ]
    elif evidence_enabled:
        sections = [
            "以下是任务约束：",
            "1. 优先依据图像和OCR证据回答。",
            "2. 如果版式分析提示为表格或图表，请优先从结构和数字中读取信息。",
            "3. 如果问题需要读图中的文字或表格，请优先使用OCR结果。",
            "4. 如果证据不足，请明确说明无法判断，并指出缺少什么证据。",
            "",
            analysis_block,
            "",
            ocr_block,
            "",
            "历史对话：",
            history_block if history_block else "无",
            "",
            f"当前问题：{question}",
        ]
    else:
        sections = [
            "请根据图像内容回答问题。",
            "",
            analysis_block,
            "",
            ocr_block,
            "",
            "历史对话：",
            history_block if history_block else "无",
            "",
            f"当前问题：{question}",
        ]
    return "\n".join(sections)
