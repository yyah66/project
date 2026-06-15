from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from src.assistant import AssistantEngine, ConversationTurn
from src.config import AppConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEMO_IMAGE_DIR = PROJECT_ROOT / "data" / "demo" / "images"


st.set_page_config(
    page_title="中文图文问答助手",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f5efe7;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: #ffffff;
            --text: #1f2937;
            --muted: #5b6472;
            --accent: #0f766e;
            --accent-soft: #ccfbf1;
            --border: rgba(31, 41, 55, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(217, 119, 6, 0.12), transparent 24%),
                linear-gradient(180deg, #fbf7f2 0%, #f3eee7 100%);
            color: var(--text);
        }

        .hero {
            padding: 1.3rem 1.4rem;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,255,255,0.68));
            border-radius: 22px;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
        }

        .hero p {
            margin: 0.45rem 0 0;
            color: var(--muted);
            font-size: 0.98rem;
        }

        .metric-card {
            padding: 1rem 1.05rem;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.84);
            box-shadow: 0 8px 26px rgba(15, 23, 42, 0.05);
        }

        .metric-title {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.2rem;
        }

        .metric-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text);
        }

        .chat-box {
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1rem 0.4rem;
            background: rgba(255,255,255,0.82);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }

        .evidence-card {
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.86);
            margin-bottom: 0.55rem;
        }

        .small-label {
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }

        .block-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(15,118,110,0.22), transparent);
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def image_to_bytes(uploaded_file) -> bytes:
    if uploaded_file is None:
        return b""
    return uploaded_file.getvalue()


def reset_conversation() -> None:
    st.session_state.messages = []


def ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("current_image_bytes", b"")
    st.session_state.setdefault("current_image_name", "")
    st.session_state.setdefault("current_image", None)
    st.session_state.setdefault("last_ocr", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_analysis", None)
    st.session_state.setdefault("last_preprocess", None)


def load_demo_sample(image_name: str, question: str) -> None:
    image_path = DEMO_IMAGE_DIR / image_name
    if not image_path.exists():
        st.warning(f"找不到演示图片：{image_name}")
        return

    image_bytes = image_path.read_bytes()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.session_state.current_image = image
    st.session_state.current_image_bytes = image_bytes
    st.session_state.current_image_name = image_path.name
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.last_ocr = None
    st.session_state.last_analysis = None
    st.session_state.last_preprocess = None
    st.session_state.example_question = question
    st.session_state.chat_input = question


@st.cache_resource
def get_engine(config: AppConfig) -> AssistantEngine:
    return AssistantEngine(config)


@st.cache_data(show_spinner="正在分析图片，请稍候...")
def inspect_image_cached(image_bytes: bytes, _config_hash: str):
    """对图片做 OCR+预处理并缓存结果。_config_hash 用于区分不同 OCR 配置下的缓存键。"""
    config = AppConfig()
    engine = AssistantEngine(config)
    return engine.inspect_image(image_bytes)


def render_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)


def build_runtime_config() -> AppConfig:
    """优先使用会话中存储的 provider / model / api_key，其次回退到环境变量/默认值。"""
    config = load_config()
    config.provider = st.session_state.get("runtime_provider", config.provider)
    config.model = st.session_state.get("runtime_model", config.model)
    config.api_key = st.session_state.get("runtime_api_key") or config.api_key
    config.lora_path = st.session_state.get("runtime_lora_path") or config.lora_path
    return config


def main() -> None:
    inject_styles()
    ensure_state()
    st.session_state.setdefault("runtime_provider", "dashscope")
    st.session_state.setdefault("runtime_model", "qwen3-vl-flash")
    st.session_state.setdefault("runtime_api_key", "")
    st.session_state.setdefault("runtime_lora_path", "")

    config = build_runtime_config()
    engine = get_engine(config)

    if (
        st.session_state.current_image is None
        and st.session_state.current_image_name == ""
        and not st.session_state.messages
        and st.session_state.last_result is None
        and st.session_state.last_ocr is None
    ):
        demo_image_path = DEMO_IMAGE_DIR / "chart_sample.png"
        if demo_image_path.exists():
            load_demo_sample("chart_sample.png", "图表整体是增长还是下降？最高季度是哪一个？")
        else:
            st.warning("演示图片尚未生成，请运行 `python scripts/generate_demo_assets.py` 生成，或直接上传图片。")

    st.markdown(
        """
        <div class="hero">
            <h1>中文图文问答助手</h1>
            <p>面向中文图文混合场景，支持图片上传、OCR证据增强、多轮对话和可追溯回答。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("运行配置")

        # --- 推理后端 ---
        provider_options = ["dashscope", "local", "heuristic"]
        provider_labels = {"dashscope": "☁️ DashScope 云端", "local": "💻 本地模型", "heuristic": "🔧 本地兜底 (无API)"}
        def _provider_index(current: str) -> int:
            try:
                return provider_options.index(current)
            except ValueError:
                return 0
        selected_provider = st.selectbox(
            "推理后端",
            provider_options,
            index=_provider_index(config.provider),
            format_func=lambda x: provider_labels.get(x, x),
            key="provider_select",
        )
        if selected_provider != st.session_state.runtime_provider:
            st.session_state.runtime_provider = selected_provider
            st.cache_resource.clear()
            st.rerun()

        # --- API Key ---
        new_key = st.text_input(
            "DashScope API Key",
            type="password",
            value=st.session_state.runtime_api_key,
            placeholder="sk-...（云端推理必需）",
            key="api_key_input",
        )
        if new_key != st.session_state.runtime_api_key:
            st.session_state.runtime_api_key = new_key
            # 填入百炼 API Key 后自动切换为云端推理
            if new_key and st.session_state.runtime_provider == "heuristic":
                st.session_state.runtime_provider = "dashscope"
            st.cache_resource.clear()
            st.rerun()

        # --- 模型选择 ---
        model_options = ["qwen3-vl-flash", "qwen3-vl-8b-instruct", "qwen2.5-vl-7b-instruct", "glm-4v-flash"]
        selected_model = st.selectbox(
            "模型",
            model_options,
            index=model_options.index(config.model) if config.model in model_options else 0,
            key="model_select",
        )
        if selected_model != st.session_state.runtime_model:
            st.session_state.runtime_model = selected_model
            # 选择云端模型时自动切 dashscope
            if st.session_state.runtime_provider == "heuristic" and st.session_state.runtime_api_key:
                st.session_state.runtime_provider = "dashscope"
            st.cache_resource.clear()
            st.rerun()

        # 无 API Key 时醒目警告
        if config.provider == "dashscope" and not config.api_key:
            st.error("⚠️ 未配置百炼 API Key，请在下方输入或设置 DASHSCOPE_API_KEY 环境变量")

        # LoRA adapter 路径（仅 local 模式）
        if selected_provider == "local":
            new_lora = st.text_input(
                "LoRA Adapter 路径",
                value=st.session_state.runtime_lora_path,
                placeholder="outputs/lora/chartqa/final_adapter（留空则使用基座模型）",
                key="lora_path_input",
            )
            if new_lora != st.session_state.runtime_lora_path:
                st.session_state.runtime_lora_path = new_lora
                st.cache_resource.clear()
                st.rerun()

        st.caption(f"OCR行数上限：{config.max_ocr_lines}")
        st.caption(f"历史轮次上限：{config.max_history_turns}")
        st.divider()
        uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg", "bmp", "webp"])
        if st.button("清空对话", use_container_width=True):
            reset_conversation()
            st.session_state.last_result = None
            st.session_state.last_ocr = None
            st.session_state.last_analysis = None
            st.session_state.last_preprocess = None
            st.session_state.current_image = None
            st.session_state.current_image_bytes = b""
            st.session_state.current_image_name = ""
            st.rerun()
        st.divider()
        st.markdown("### 示例问题")
        examples = [
            "这张图的主要内容是什么？",
            "图中的标题是什么？",
            "请指出图片里最关键的文字证据。",
            "如果这是表格或图表，请解释其中的数字或趋势。",
        ]
        for example in examples:
            if st.button(example, key=f"example-{example}"):
                st.session_state.example_question = example

        st.divider()
        st.markdown("### 一键演示样例")
        demo_buttons = [
            ("文档样例", "document_sample.png", "这张图的主题是什么？"),
            ("表格样例", "table_sample.png", "哪种设置的准确率最高？"),
            ("图表样例", "chart_sample.png", "图表整体是增长还是下降？最高季度是哪一个？"),
        ]
        for label, filename, question in demo_buttons:
            if st.button(label, key=f"demo-{filename}"):
                load_demo_sample(filename, question)
                st.rerun()

    if uploaded_file is not None:
        image_bytes = image_to_bytes(uploaded_file)
        if image_bytes and uploaded_file.name != st.session_state.current_image_name:
            config_hash = config.model + config.provider + (config.api_key[:8] if config.api_key else "")
            image, preprocess_result, ocr_result = inspect_image_cached(image_bytes, config_hash)
            st.session_state.current_image = image
            st.session_state.current_image_bytes = image_bytes
            st.session_state.current_image_name = uploaded_file.name
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.last_ocr = ocr_result
            st.session_state.last_analysis = ocr_result.analysis if ocr_result else None
            st.session_state.last_preprocess = preprocess_result

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("<div class='metric-card'><div class='metric-title'>当前图片</div><div class='metric-value'>" + (st.session_state.current_image_name or "尚未上传") + "</div></div>", unsafe_allow_html=True)
        st.write("")

        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image)
        else:
            st.info("请先在左侧上传图片，然后输入中文问题开始对话。")

        st.markdown("<div class='block-divider'></div>", unsafe_allow_html=True)

        if st.session_state.last_result is not None:
            result = st.session_state.last_result
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">模型回答</div>
                    <div class="metric-value">{result.answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            cols = st.columns(3)
            cols[0].metric("置信度", result.confidence)
            cols[1].metric("推理来源", result.provider)
            cols[2].metric("OCR后端", st.session_state.last_ocr.backend if st.session_state.last_ocr else "none")

            if st.session_state.last_analysis is not None:
                analysis = st.session_state.last_analysis
                analysis_cols = st.columns(4)
                analysis_cols[0].metric("场景类型", analysis.scene_type)
                analysis_cols[1].metric("文本密度", analysis.text_density)
                analysis_cols[2].metric("表格", "是" if analysis.likely_table else "否")
                analysis_cols[3].metric("图表", "是" if analysis.likely_chart else "否")

            if st.session_state.last_preprocess is not None:
                preprocess = st.session_state.last_preprocess
                preprocess_cols = st.columns(4)
                preprocess_cols[0].metric("原始尺寸", f"{preprocess.original_size[0]}x{preprocess.original_size[1]}")
                preprocess_cols[1].metric("处理后尺寸", f"{preprocess.processed_size[0]}x{preprocess.processed_size[1]}")
                preprocess_cols[2].metric("旋转校正", f"{preprocess.rotation_applied}°")
                preprocess_cols[3].metric("尺寸归一", "是" if preprocess.resized else "否")

            if result.evidence:
                st.markdown("### 证据引用")
                for item in result.evidence:
                    st.markdown(f"<div class='evidence-card'>{item}</div>", unsafe_allow_html=True)
            if result.uncertainty:
                st.warning(result.uncertainty)

        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            render_message(message.role, message.content)
        st.markdown("</div>", unsafe_allow_html=True)

        # 处理侧边栏示例按钮设置的预填问题
        prefill = st.session_state.pop("example_question", None)
        question = st.chat_input(
            placeholder="例如：这张图里最关键的信息是什么？",
            key="chat_input",
            disabled=st.session_state.current_image is None,
        )
        if prefill and not question:
            question = prefill

        if question and st.session_state.current_image is not None:
            user_turn = ConversationTurn(role="user", content=question)
            st.session_state.messages.append(user_turn)

            with st.spinner("正在分析图像和生成回答..."):
                image = st.session_state.current_image
                image_bytes = st.session_state.current_image_bytes
                ocr_result = st.session_state.last_ocr or engine.ocr_service.extract(image)
                result = engine.answer_question(
                    image=image,
                    image_bytes=image_bytes,
                    question=question,
                    history=st.session_state.messages[:-1],
                    ocr_result=ocr_result,
                )
                st.session_state.last_result = result
                st.session_state.last_ocr = ocr_result

            # 对话历史只保存纯净回答，证据和不确定信息在侧边栏单独展示
            assistant_turn = ConversationTurn(role="assistant", content=result.answer)
            st.session_state.messages.append(assistant_turn)
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">系统说明</div>
                <div class="metric-value">证据优先 · 低幻觉 · 可追溯</div>
                <p style="color: var(--muted); margin: 0.5rem 0 0; line-height: 1.7;">
                本原型会先提取 OCR，再把问题、OCR 和历史对话一起送入视觉语言模型。
                如果没有配置 API Key，会退化到保守兜底模式，仍然保留完整的交互流程。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        if st.session_state.last_ocr is not None:
            ocr_result = st.session_state.last_ocr
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">OCR状态</div>
                    <div class="metric-value">{ocr_result.backend}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if ocr_result.lines:
                st.markdown("### OCR片段")
                for index, line in enumerate(ocr_result.lines[:10], start=1):
                    bbox = f" | bbox={line.bbox}" if line.bbox else ""
                    score = f" | score={line.confidence:.3f}" if line.confidence is not None else ""
                    st.markdown(
                        f"<div class='evidence-card'><div class='small-label'>#{index}{bbox}{score}</div><div>{line.text}</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("当前未提取到 OCR 文本。若图片是密集文字，建议安装并启用 OCR 后端。")
        elif st.session_state.current_image is not None:
            st.info("图片已加载，OCR 会在你提问时再执行，以便首屏更快打开。")
        else:
            st.info("上传图片或选择示例后，OCR 会按需执行。")

        if st.session_state.last_analysis is not None:
            analysis = st.session_state.last_analysis
            st.markdown("### 版式分析")
            st.markdown(
                f"<div class='evidence-card'><div class='small-label'>场景</div><div>{analysis.scene_type}</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='evidence-card'><div class='small-label'>文本密度</div><div>{analysis.text_density}</div></div>",
                unsafe_allow_html=True,
            )
            note_text = "；".join(analysis.notes) if analysis.notes else "无"
            st.markdown(
                f"<div class='evidence-card'><div class='small-label'>提示</div><div>{note_text}</div></div>",
                unsafe_allow_html=True,
            )

        if st.session_state.last_preprocess is not None:
            preprocess = st.session_state.last_preprocess
            st.markdown("### 预处理")
            st.markdown(
                f"<div class='evidence-card'><div class='small-label'>步骤</div><div>{'；'.join(preprocess.steps)}</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='evidence-card'><div class='small-label'>尺寸变化</div><div>{preprocess.original_size[0]}x{preprocess.original_size[1]} → {preprocess.processed_size[0]}x{preprocess.processed_size[1]}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### 实验建议")
        st.write("1. 零样本基线：直接使用当前原型回答。")
        st.write("2. OCR增强：对比启用和关闭 OCR 的回答差异。")
        st.write("3. 证据提示：观察是否更少出现看图乱答。")
        st.write("4. 模型对照：切换不同 VLM 的效果。")


if __name__ == "__main__":
    main()