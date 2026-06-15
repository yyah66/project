# OCR + Evidence 增强重训指南

## 背景

当前训练脚本使用简单 prompt（纯文本问答），与 `app.py` 推理时的 OCR 增强 + JSON 输出格式不一致。本次重训将训练 prompt 对齐到推理格式，并扩充数据集至 ~5000 条。

## 数据现状

| 数据集 | 条数 | 图片 | Evidence 自带 |
|---|---|---|---|
| DocVQA (HF) | 1000 | HF 内嵌 | 待确认 |
| VQA-v2 | 2000 | COCO val2014, 1931 张 | 无 |
| TextVQA | 2000 | OpenImages, 1706 张 | ocr_tokens |
| ChartQA | 50 | 已有 | 空 |
| 自建中文 | 122 | 已有 | 有 |
| **合计** | **~5172** | — | — |

## 新增/修改文件

| 文件 | 说明 |
|---|---|
| `scripts/check_gpu_ocr.py` | **新建** — GPU OCR 兼容性诊断，运行前执行 |
| `scripts/precompute_all_ocr.py` | **修改** — 新增 `--gpu` GPU 加速模式 + 断点续跑 |
| `scripts/train_qwen2vl_lora.py` | **修改** — 支持 OCR+JSON prompt，max_length 升至 4096 |
| `scripts/evaluate.py` | **重写** — 支持 `--ablation` 一键 8 组消融 |
| `src/prompt.py` | 不动（复用） |
| `scripts/setup_autodl.sh` | **新建** — AutoDL 数据盘初始化（HF_HOME + PIP 缓存重定向） |
| `scripts/download_easyocr_models.py` | **新建** — EasyOCR 小模型下载辅助（国内 HF 镜像） |
| `src/ocr.py` | 不动（复用） |

## 前置准备：数据盘配置

**AutoDL 系统盘约 50GB，HuggingFace 模型（~15GB）必须放数据盘。OCR 小模型（~100MB）走默认路径即可。**

```bash
source scripts/setup_autodl.sh
```

验证：

```bash
echo $HF_HOME       # 应输出 /root/autodl-tmp/cache/huggingface
echo $PIP_CACHE_DIR # 应输出 /root/autodl-tmp/cache/pip
```

> 新终端需 `source ~/.bashrc`。脚本只需执行一次。

## 执行流程

### Step 0: GPU 环境验证

**此步骤必须在 OCR 预计算前执行**，确认 GPU OCR 后端可用。

```bash
python3 scripts/check_gpu_ocr.py
```

此脚本依次检查：
1. CUDA / PyTorch GPU 基础环境
2. PaddlePaddle GPU 是否可用
3. PaddleOCR GPU 推理速度（3 轮热身测试）
4. EasyOCR GPU（备选方案）
5. RapidOCR CPU 基线速度

输出示例：
```
=== 总结与建议 ===
  环境: NVIDIA GeForce RTX 5090 (31.4 GB, sm_12.0)

  *** 使用 EasyOCR GPU ***
  单张速度: 0.52s（预热后 ~0.29s）
  预计 5000 张耗时: ~25-40 分钟

  运行命令:
    python3 scripts/precompute_all_ocr.py --gpu
```

**常见问题与对策：**

| 诊断结果 | 原因 | 对策 |
|---|---|---|
| PaddlePaddle 未安装 | 未预装 | `pip install paddlepaddle-gpu`（注意：2.6.2 暂不支持 Blackwell） |
| PaddleOCR 初始化失败 | PaddlePaddle 2.6.2 不支持 Blackwell (sm_12.0) | 使用 EasyOCR（已确认可用） |
| EasyOCR 模型下载超时 | GitHub Releases 国内不通 | 从本机 `~/.EasyOCR/model/` 上传 `.pth` 文件到 AutoDL，见下方说明 |
| CUDA 不可用 | AutoDL 镜像不含 GPU 驱动 | 重启实例选择含 CUDA 12.8+ 的镜像 |

**EasyOCR 模型上传（网络不通时）：**

```bash
# 在本机 Windows 下载模型
pip install easyocr
python -c "import easyocr; r = easyocr.Reader(['ch_sim','en'], gpu=False); print('done')"

# 上传以下文件到 AutoDL（通过网页控制台文件管理）
# C:\Users\<用户名>\.EasyOCR\model\*.pth → /root/.EasyOCR/model/

# 在 AutoDL 确认
ls -lh /root/.EasyOCR/model/
# craft_mlt_25k.pth  (~83MB)
# zh_sim_g2.pth      (~22MB)
```

> **实测环境**: PyTorch 2.8.0+cu128, CUDA 12.8, RTX 5090 31.4GB。PaddlePaddle 2.6.2 不兼容 sm_12.0，EasyOCR 单张 0.52s（预热后 0.29s），预估 5000 张 25-40 分钟。

### Step 1: 确认 DocVQA 格式

```bash
python3 scripts/check_docvqa_keys.py
```

根据输出字段名调整 DocVQA 转换脚本。完成后生成 `data/public/docvqa_1000.jsonl`。

### Step 2: OCR 预计算（GPU 加速 ≈ 10-25 分钟，原 CPU 方式 ≈ 30-60 分钟）

> **当前环境**: PaddlePaddle 2.6.2 不支持 Blackwell (sm_12.0)，**必须使用 EasyOCR**。等 PaddlePaddle 3.0+ 发布后可切换。

```bash
# EasyOCR GPU（基于 PyTorch，完全兼容 RTX 5090）
python3 scripts/precompute_all_ocr.py --gpu --gpu-backend easyocr

# 如果 EasyOCR 模型下载失败（GitHub 不通），先运行:
python3 scripts/download_easyocr_models.py
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--gpu` | 关闭 | 启用 GPU 加速模式 |
| `--gpu-backend` | `easyocr` | GPU 后端: `easyocr` (PyTorch) 或 `paddle` (PaddleOCR，需 3.0+) |
| `--max-side` | `1600` | 送入 OCR 的图片最大边长，降至 1024 可进一步提速 |
| `--max-ocr-lines` | `24` | 每张图片最多保留的 OCR 行数 |
| `--checkpoint-every` | `100` | 每 N 张图片保存一次检查点（支持断点续跑） |

#### 传统方式：CPU 多进程（无 GPU 时回退）

```bash
python3 scripts/precompute_all_ocr.py --workers 64
```

扫描 `data/public/` 和 `data/training/` 下所有 JSONL，对每张图片跑 OCR，输出：

```
data/training/merged/
  merged_train_enhanced.jsonl    # ~4000 条
  merged_val_enhanced.jsonl      # ~500 条
  merged_test_enhanced.jsonl     # ~500 条
  merged_package.json
```

每条记录新增 `ocr_lines` 和 `ocr_analysis` 字段。

#### 断点续跑

如果预计算过程中断（OOM、进程被杀等），直接重新运行相同命令即可自动续跑——已完成的图片不会被重复处理。检查点每 100 张图片自动保存。

### Step 3: 训练

训练前务必确认：
1. `HF_HOME` 已指向数据盘（模型加载不会下载到系统盘）
2. 模型已下载到数据盘 `/root/autodl-tmp/models/`

```bash
python3 scripts/train_qwen2vl_lora.py \
    --model-path /root/autodl-tmp/models/qwen/Qwen2.5-VL-7B-Instruct \
    --train-jsonl data/training/merged/merged_train_enhanced.jsonl \
    --val-jsonl data/training/merged/merged_val_enhanced.jsonl \
    --image-root . \
    --output-dir outputs/lora/merged_ocr \
    --max-length 4096 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --lora-rank 64 \
    --epochs 3
```

训练时 prompt 格式：

```
System: build_system_prompt(evidence_enabled=True)
  → 证据规则 + JSON 输出 {"answer":"...","evidence":[...],"confidence":"高","uncertainty":""}

User:   build_user_prompt(question, ocr_result, [])
  → 版式分析 + OCR 文本块 + 证据约束 + 问题 + 图片

Assistant: {"answer":"<原始答案>", "evidence":[...], "confidence":"高", "uncertainty":""}
```

### Step 4: 消融实验

```bash
python3 scripts/evaluate.py --ablation \
    --adapter outputs/lora/merged_ocr/final_adapter \
    --test-jsonl data/training/merged/merged_test_enhanced.jsonl

python3 scripts/summarize_ablation.py
```

8 组单进程跑完，输出到 `outputs/ablation/`：

| 组号 | OCR | Evidence | Model |
|---|---|---|---|
| 1 | ON | ON | LoRA |
| 2 | ON | ON | Base |
| 3 | ON | OFF | LoRA |
| 4 | ON | OFF | Base |
| 5 | OFF | ON | LoRA |
| 6 | OFF | ON | Base |
| 7 | OFF | OFF | LoRA |
| 8 | OFF | OFF | Base |

## 速度预估

| 阶段 | 原 CPU 方式 | GPU 加速 (RTX 5090) |
|---|---|---|
| Step 0: GPU 诊断 | — | ~1 分钟 |
| Step 1: DocVQA 格式 | ~1 分钟 | 不变 |
| Step 2: OCR 预计算 | 30-60 分钟 | **10-25 分钟** |
| Step 3: 训练 | ~2-4 小时 | 不变（已使用 GPU） |
| Step 4: 消融实验 | ~1-2 小时 | 不变（已使用 GPU） |

## 消融变量说明

| 变量 | ON | OFF |
|---|---|---|
| **OCR** | `OCRService` + user prompt 含 OCR 文本 | `NoOpOCRService` + 空 OCR 块 |
| **Evidence** | `build_system_prompt(True)` + JSON 输出解析 | `build_system_prompt(False)` 纯文本 |
| **Model** | LoRA adapter | `disable_adapter_layers()` 基座 |

## 故障排查

### PaddleOCR GPU 初始化失败

```bash
# 先诊断
python3 scripts/check_gpu_ocr.py

# 如果诊断显示 PaddlePaddle 不支持当前 CUDA/GPU，切换到 EasyOCR
pip install easyocr
python3 scripts/precompute_all_ocr.py --gpu --gpu-backend easyocr
```

### GPU 显存不足 (OOM)

```bash
# 降低图片分辨率减少显存占用
python3 scripts/precompute_all_ocr.py --gpu --max-side 1024
```

### 预计算中断后恢复

直接重新运行相同命令，断点续跑会自动跳过已完成的图片：

```bash
python3 scripts/precompute_all_ocr.py --gpu
# 输出: 断点续跑: 已完成 2340/5000 条
```

### 最终回退：CPU 模式

如果所有 GPU 方案都失败，仍可使用原始的 CPU 多进程模式：

```bash
python3 scripts/precompute_all_ocr.py --workers 64
```
