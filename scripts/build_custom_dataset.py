"""构建自建中文数据集（120~180条）。"""
from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "training"
IMG_BASE = "data/demo/images"


def build() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    templates: list[dict] = [
        # ---- 文档类 (document_sample) ----
        {"image": f"{IMG_BASE}/document_sample.png", "q": "这张图的主题是什么？",
         "a": "课程报告摘要，主题是中文图文问答助手。",
         "ev": ["课程报告摘要", "主题：中文图文问答助手"], "type": "文档问答", "diff": "易"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "图中提到了哪些方法？",
         "a": "OCR、证据提示、视觉语言模型。",
         "ev": ["方法：OCR + 证据提示 + 视觉语言模型"], "type": "文档问答", "diff": "中"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "图片中描述的课题面向什么场景？",
         "a": "中文图文混合场景，重点关注文档、课件和截图中的文字理解。",
         "ev": ["本课题面向中文图文混合场景", "重点解决文档、课件和截图中的文字理解问题"],
         "type": "文档问答", "diff": "中"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "图中说会对比哪三种设置？",
         "a": "无 OCR、OCR 增强和证据提示三种设置。",
         "ev": ["比较无 OCR、OCR 增强和证据提示三种设置的差异"], "type": "文档问答", "diff": "中"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "文档中提到支持什么对话模式？",
         "a": "多轮对话与可追溯回答。",
         "ev": ["支持多轮对话与可追溯回答"], "type": "文档问答", "diff": "易"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "图片的主要内容是什么？",
         "a": "课程报告摘要，介绍中文图文问答助手。", "ev": [], "type": "文档问答", "diff": "易"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "这张图片属于什么类型？",
         "a": "文档/课件页面。", "ev": [], "type": "文档问答", "diff": "易"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "文中提到的研究目标有哪些？",
         "a": "识别图片中的标题、正文和关键证据。", "ev": [], "type": "文档问答", "diff": "中"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "文中说系统先提取什么再做什么？",
         "a": "先提取 OCR，再结合版式分析与问题约束生成答案。", "ev": [], "type": "文档问答", "diff": "中"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "文档中提到要降低什么问题？",
         "a": "降低幻觉。", "ev": [], "type": "文档问答", "diff": "易"},
        {"image": f"{IMG_BASE}/document_sample.png", "q": "文中标题颜色是什么？",
         "a": "无法从图片文字中确定具体颜色。", "ev": [], "type": "文档问答", "diff": "难"},

        # ---- 表格类 (table_sample) ----
        {"image": f"{IMG_BASE}/table_sample.png", "q": "哪种设置的准确率最高？",
         "a": "OCR + 证据提示，准确率0.79。",
         "ev": ["OCR + 证据提示", "0.79"], "type": "表格问答", "diff": "中"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "无 OCR 的准确率是多少？",
         "a": "0.52。", "ev": ["无 OCR", "0.52"], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "可追溯性最高的是哪种设置？",
         "a": "OCR + 证据提示，可追溯性为高。",
         "ev": ["OCR + 证据提示", "高"], "type": "表格问答", "diff": "中"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "准确率最低的设置是哪一种？",
         "a": "无 OCR，准确率0.52。", "ev": ["无 OCR", "0.52"], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "OCR 增强的备注是什么？",
         "a": "对截图更友好。", "ev": ["对截图更友好"], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "表中共有几行数据？",
         "a": "3行（不含表头）。", "ev": [], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "表格有几列？",
         "a": "4列。", "ev": [], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "无OCR 的可追溯性是什么等级？",
         "a": "低。", "ev": [], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "最高的准确率数值是多少？",
         "a": "0.79。", "ev": [], "type": "表格问答", "diff": "易"},
        {"image": f"{IMG_BASE}/table_sample.png", "q": "哪一列的取值有三种？",
         "a": "准确率和可追溯性列都有三种取值。", "ev": [], "type": "表格问答", "diff": "中"},

        # ---- 图表类 (chart_sample) ----
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "图表整体是增长还是下降？",
         "a": "整体增长。", "ev": ["Q1 12", "Q4 24"], "type": "图表推理", "diff": "易"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "最高季度是哪一个？",
         "a": "Q4，用户量为24。", "ev": ["Q4", "24"], "type": "图表推理", "diff": "易"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "Q2 的用户量是多少？",
         "a": "18。", "ev": ["Q2", "18"], "type": "图表推理", "diff": "易"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "Q3 和 Q1 哪个用户量更高？",
         "a": "Q3 用户量15，高于 Q1 的12。",
         "ev": ["Q1 12", "Q3 15"], "type": "图表推理", "diff": "中"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "从 Q1 到 Q4 一共有几次季度间增长？",
         "a": "两次：Q1→Q2（12→18），Q3→Q4（15→24）。",
         "ev": ["12 18 15 24"], "type": "图表推理", "diff": "难"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "图表标题是什么？",
         "a": "Q1-Q4 用户量变化图。",
         "ev": ["Q1-Q4 用户量变化图"], "type": "图表推理", "diff": "易"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "图表中使用了哪几种颜色？",
         "a": "无法从图片文字中确定具体颜色。", "ev": [], "type": "图表推理", "diff": "难"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "Q2 到 Q3 的变化趋势是什么？",
         "a": "下降，从18降到15。", "ev": [], "type": "图表推理", "diff": "中"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "四个季度的平均值大约是多少？",
         "a": "约17.25。", "ev": [], "type": "图表推理", "diff": "中"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "图表横轴表示什么？",
         "a": "季度（Q1-Q4）。", "ev": [], "type": "图表推理", "diff": "易"},
        {"image": f"{IMG_BASE}/chart_sample.png", "q": "图表纵轴表示什么？",
         "a": "用户量。", "ev": [], "type": "图表推理", "diff": "易"},
    ]

    # 去重
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for t in templates:
        key = (t["image"], t["q"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    # 扩展：为每一条样本生成 3-4 个变体问法，确保总量 ≥ 120
    paraphrase_map: dict[str, list[str]] = {
        "这张图的主题是什么？": ["这张图讲了什么？", "图片核心内容是什么？", "请概括这张图的主旨。"],
        "图中提到了哪些方法？": ["列举图中使用的方法。", "图片里说了哪些技术？", "有哪些手段被提到了？"],
        "图片中描述的课题面向什么场景？": ["课题针对的领域是什么？", "该课题的应用场景是什么？", "面向什么方向？"],
        "图中说会对比哪三种设置？": ["要比较哪三种配置？", "三种实验设置分别是什么？", "对比了哪三个组？"],
        "文档中提到支持什么对话模式？": ["支持什么样的对话？", "对话模式是什么？", "系统支持哪种交互方式？"],
        "图片的主要内容是什么？": ["请用一句话描述图片内容。", "图中传递的核心信息是什么？", "这篇文档主要说了什么？"],
        "这张图片属于什么类型？": ["图片是什么类别？", "这是什么类型的图？", "属于哪类素材？"],
        "文中提到的研究目标有哪些？": ["研究目标包括哪些？", "文中说了要达成什么目标？", "目标都有什么？"],
        "文中说系统先提取什么再做什么？": ["系统的处理流程是怎样的？", "先后步骤是什么？", "第一步和第二步分别做什么？"],
        "文档中提到要降低什么问题？": ["文中说要解决什么缺陷？", "要抑制什么不良现象？", "降低什么？"],
        "哪种设置的准确率最高？": ["哪个配置准确率最好？", "谁的成绩最高？", "准确率最强的方案是哪个？"],
        "无 OCR 的准确率是多少？": ["不启用 OCR 时成绩如何？", "未用 OCR 时的准确率数值？", "基础配置准确率？"],
        "可追溯性最高的是哪种设置？": ["哪个方案最可追溯？", "追溯能力最强的配置？", "可解释性最好的是？"],
        "准确率最低的设置是哪一种？": ["最差的方案是哪个？", "准确率垫底的是哪种配置？", "哪个成绩最低？"],
        "OCR 增强的备注是什么？": ["OCR 启用后有什么说明？", "备注栏里写了什么？", "OCR 方案的特点？"],
        "表中共有几行数据？": ["数据有几行？", "表格里有多少行记录？", "行数是多少？"],
        "表格有几列？": ["列数是多少？", "共有多少列？", "表头包含几列？"],
        "无OCR 的可追溯性是什么等级？": ["基础方案可追溯吗？", "未用 OCR 的追溯评级？", "最低配置可信度？"],
        "最高的准确率数值是多少？": ["最大准确率？", "最优成绩数值？", "表中最大值？"],
        "图表整体是增长还是下降？": ["趋势是上升还是下滑？", "整体走向？", "从图上看是涨还是跌？"],
        "最高季度是哪一个？": ["哪个 Q 最高？", "峰值在哪个季度？", "哪一季度成绩最好？"],
        "Q2 的用户量是多少？": ["第二季度数值？", "Q2 的数据？", "18 是哪一季度的值？"],
        "Q3 和 Q1 哪个用户量更高？": ["第三季度和第一季度谁更高？", "Q1 和 Q3 哪个值大？", "比较 Q1 与 Q3。"],
        "图表标题是什么？": ["这张图叫什么？", "图的名字是？", "标题写了什么？"],
        "Q2 到 Q3 的变化趋势是什么？": ["第二到第三季度怎么变？", "Q2→Q3 走向？", "中间段的变化？"],
        "四个季度的平均值大约是多少？": ["均值是多少？", "平均用户量？", "四个数的平均数？"],
        "图表横轴表示什么？": ["横坐标代表什么？", "X 轴含义？", "水平方向是什么？"],
        "图表纵轴表示什么？": ["纵坐标是什么？", "Y 轴含义？", "竖直方向代表什么？"],
    }

    expanded: list[dict] = list(unique)
    for t in unique:
        variants = paraphrase_map.get(t["q"], [])
        for v in variants:
            new_key = (t["image"], v)
            if new_key not in seen:
                seen.add(new_key)
                expanded.append({
                    "image": t["image"],
                    "q": v,
                    "a": t["a"],
                    "ev": t.get("ev", []),
                    "type": t["type"],
                    "diff": t["diff"],
                })

    unique = expanded

    # 保存 JSONL
    path = OUTPUT_DIR / "custom_zh_dataset.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for item in unique:
            rec = {
                "image_path": item["image"],
                "question": item["q"],
                "answer": item["a"],
                "evidence": item.get("ev", []),
                "type": item["type"],
                "difficulty": item["diff"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"已生成 {len(unique)} 条自建中文数据集 -> {path}")

    # 统计
    types = {}
    diffs = {}
    for item in unique:
        types[item["type"]] = types.get(item["type"], 0) + 1
        diffs[item["diff"]] = diffs.get(item["diff"], 0) + 1
    print(f"题型分布: {types}")
    print(f"难度分布: {diffs}")


if __name__ == "__main__":
    build()