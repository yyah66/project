from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"
IMAGES_DIR = DEMO_DIR / "images"


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill: str = "#111827") -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=6)


def create_document_image(path: Path) -> None:
    img = Image.new("RGB", (1280, 900), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(40)
    body_font = get_font(28)
    small_font = get_font(22)

    draw.rounded_rectangle((60, 50, 1220, 840), radius=26, outline="#d1d5db", width=3, fill="#fcfcfd")
    draw.text((110, 100), "课程报告摘要", font=title_font, fill="#0f766e")
    draw.text((110, 170), "主题：中文图文问答助手", font=body_font, fill="#111827")
    draw.text((110, 235), "任务：识别图片中的标题、正文和关键证据", font=body_font, fill="#111827")
    draw.text((110, 300), "方法：OCR + 证据提示 + 视觉语言模型", font=body_font, fill="#111827")
    draw.text((110, 365), "结果：支持多轮对话与可追溯回答", font=body_font, fill="#111827")

    draw.rounded_rectangle((110, 470, 1160, 710), radius=18, outline="#94a3b8", width=2, fill="#f8fafc")
    paragraphs = [
        "本课题面向中文图文混合场景，重点解决文档、课件和截图中的文字理解问题。",
        "系统会先提取 OCR，再结合版式分析与问题约束生成答案，从而降低幻觉。",
        "实验部分将比较无 OCR、OCR 增强和证据提示三种设置的差异。",
    ]
    y = 500
    for paragraph in paragraphs:
        draw.text((145, y), paragraph, font=small_font, fill="#1f2937")
        y += 58

    draw.text((110, 760), "图注：文档类样例", font=small_font, fill="#6b7280")
    img.save(path)


def create_table_image(path: Path) -> None:
    img = Image.new("RGB", (1280, 900), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42)
    cell_font = get_font(28)
    small_font = get_font(22)

    draw.rounded_rectangle((50, 40, 1230, 850), radius=26, outline="#cbd5e1", width=3, fill="#f8fafc")
    draw.text((100, 90), "实验结果表", font=title_font, fill="#0f766e")
    draw.text((100, 150), "模块对比：是否引入 OCR 与证据提示", font=small_font, fill="#475569")

    x0, y0 = 100, 240
    col_widths = [360, 240, 240, 240]
    row_height = 110
    headers = ["设置", "准确率", "可追溯性", "备注"]
    rows = [
        ["无 OCR", "0.52", "低", "容易漏读小字"],
        ["OCR 增强", "0.68", "中", "对截图更友好"],
        ["OCR + 证据提示", "0.79", "高", "最适合演示"],
    ]

    def draw_cell(x: int, y: int, w: int, h: int, text: str, fill: str, outline: str = "#94a3b8", font=None) -> None:
        draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=2)
        if font is None:
            font = cell_font
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
        tx = x + (w - (bbox[2] - bbox[0])) / 2
        ty = y + (h - (bbox[3] - bbox[1])) / 2
        draw.multiline_text((tx, ty), text, font=font, fill="#111827", spacing=4, align="center")

    x = x0
    for header, width in zip(headers, col_widths):
        draw_cell(x, y0, width, row_height, header, "#dbeafe", font=cell_font)
        x += width

    for row_index, row in enumerate(rows, start=1):
        x = x0
        fill = "#ffffff" if row_index % 2 else "#f8fafc"
        for item, width in zip(row, col_widths):
            draw_cell(x, y0 + row_height * row_index, width, row_height, item, fill, font=cell_font)
            x += width

    draw.text((100, 620), "图注：表格类样例", font=small_font, fill="#6b7280")
    img.save(path)


def create_chart_image(path: Path) -> None:
    img = Image.new("RGB", (1280, 900), "#fefefe")
    draw = ImageDraw.Draw(img)
    title_font = get_font(42)
    label_font = get_font(26)
    small_font = get_font(22)

    draw.rounded_rectangle((50, 40, 1230, 850), radius=26, outline="#d1d5db", width=3, fill="#ffffff")
    draw.text((100, 90), "Q1-Q4 用户量变化图", font=title_font, fill="#0f766e")
    draw.text((100, 150), "问题示例：今年用户量是增长还是下降？最高季度是哪一个？", font=small_font, fill="#475569")

    chart_left, chart_top, chart_right, chart_bottom = 150, 250, 1140, 760
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline="#94a3b8", width=2, fill="#f8fafc")
    draw.line((chart_left + 30, chart_bottom - 40, chart_right - 30, chart_bottom - 40), fill="#64748b", width=3)
    draw.line((chart_left + 30, chart_top + 20, chart_left + 30, chart_bottom - 40), fill="#64748b", width=3)

    values = [12, 18, 15, 24]
    labels = ["Q1", "Q2", "Q3", "Q4"]
    colors = ["#0f766e", "#38bdf8", "#f59e0b", "#ef4444"]
    max_value = max(values)
    bar_area_height = (chart_bottom - chart_top) - 90
    bar_width = 140
    gap = 95
    start_x = chart_left + 90
    for index, (value, label, color) in enumerate(zip(values, labels, colors)):
        bar_height = int(bar_area_height * value / max_value)
        x1 = start_x + index * (bar_width + gap)
        y1 = chart_bottom - 40 - bar_height
        x2 = x1 + bar_width
        y2 = chart_bottom - 40
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=color)
        draw.text((x1 + 35, y1 - 35), str(value), font=label_font, fill="#111827")
        draw.text((x1 + 35, chart_bottom - 25), label, font=label_font, fill="#334155")

    draw.text((100, 790), "图注：图表类样例", font=small_font, fill="#6b7280")
    img.save(path)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    create_document_image(IMAGES_DIR / "document_sample.png")
    create_table_image(IMAGES_DIR / "table_sample.png")
    create_chart_image(IMAGES_DIR / "chart_sample.png")

    records = [
        {
            "image_path": "data/demo/images/document_sample.png",
            "question": "这张图的主题是什么？",
            "answer": "课程报告摘要，主题是中文图文问答助手。",
            "evidence": ["课程报告摘要", "主题：中文图文问答助手"],
            "type": "文档问答",
            "difficulty": "易",
        },
        {
            "image_path": "data/demo/images/table_sample.png",
            "question": "哪种设置的准确率最高？",
            "answer": "OCR + 证据提示，准确率最高，为 0.79。",
            "evidence": ["OCR + 证据提示", "0.79"],
            "type": "表格问答",
            "difficulty": "中",
        },
        {
            "image_path": "data/demo/images/chart_sample.png",
            "question": "图表整体是增长还是下降？最高季度是哪一个？",
            "answer": "整体增长，最高季度是 Q4。",
            "evidence": ["Q1 12", "Q2 18", "Q3 15", "Q4 24"],
            "type": "图表推理",
            "difficulty": "中",
        },
    ]

    jsonl_path = DEMO_DIR / "demo_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    readme_path = DEMO_DIR / "README.md"
    readme_path.write_text(
        "# 演示样例\n\n"
        "这里包含三类可直接上传的合成样例：文档、表格和图表。\n\n"
        "## 文件\n\n"
        "- `images/document_sample.png`\n"
        "- `images/table_sample.png`\n"
        "- `images/chart_sample.png`\n"
        "- `demo_dataset.jsonl`\n\n"
        "## 评测\n\n"
        "```bash\n"
        "python scripts/evaluate_dataset.py --input data/demo/demo_dataset.jsonl --output outputs/demo_predictions.jsonl\n"
        "```\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()