好的，这是一个结合 `CacheBlend` 论文和其 `LMCache` GitHub 仓库的实验复现指导文件。

本文档旨在帮助你一步步地复现 [CacheBlend: A General, High-Performance, and Cost-Effective Approach for Large Language Model Serving with Retrieval Augmentation](https://arxiv.org/html/2405.16444v3) 论文中提到的所有相关实验。

-----

### **复现指南：CacheBlend 实验**

#### **1. 准备工作：环境与依赖**

在开始之前，请确保你的实验环境满足以下要求：

  * **操作系统:** Linux
  * **Python 版本:** 3.8+
  * **硬件:**
      * 至少一块 NVIDIA GPU (推荐 Ampere 架构或更新的显卡，如 A100 或 H100，以获得最佳性能)。
      * 足够的内存以加载大型语言模型和数据集。
  * **软件依赖:**
      * PyTorch 2.1.0+
      * Hugging Face Transformers
      * CUDA Toolkit

**步骤 1：克隆 LMCache 仓库**

首先，将 `LMCache` 的官方 GitHub 仓库克隆到你的本地环境中：

```bash
git clone https://github.com/LMCache/LMCache.git
cd LMCache
```

**步骤 2：安装依赖**

`LMCache` 仓库提供了一个 `requirements.txt` 文件，包含了所有必要的 Python 依赖。你可以使用 `pip` 进行安装：

```bash
pip install -r requirements.txt
```

#### **2. 复现端到端 RAG 性能实验**

这部分实验旨在评估 `CacheBlend` 在完整 RAG 流程中的延迟和准确性。

**步骤 1：下载数据集和模型**

  * **数据集:** 论文中使用了 2WikiMQA、MusiQue 和 LongBench 等数据集。你需要根据 `LMCache` 仓库中的说明或脚本来下载和预处理这些数据集。通常，在仓库的 `scripts` 或 `data` 目录下可以找到相关的处理脚本。

  * **模型:** 实验基于 Llama-7B 模型。你需要从 Hugging Face Hub 下载 Llama-7B 的权重。请确保你有访问该模型的权限。

**步骤 2：运行实验脚本**

`LMCache` 仓库的 `scripts` 目录下应该包含了运行端到端 RAG 实验的脚本。你需要仔细阅读这些脚本，并根据你的环境配置（如模型路径、数据集路径、GPU ID 等）进行修改。

一个典型的实验运行命令可能如下所示 (请根据实际脚本进行调整):

```bash
python scripts/run_rag_experiment.py \
    --model_name_or_path /path/to/your/llama-7b \
    --dataset_name 2wikimqa \
    --output_dir ./results/2wikimqa \
    --use_cacheblend \
    --batch_size 8
```

**步骤 3：复现基线对比**

为了与论文中的基线（如标准 RAG、SGLang、RAGCache）进行比较，你需要：

  * **标准 RAG:** 运行不带 `CacheBlend` 优化的实验脚本。通常，脚本中会有禁用缓存的选项 (例如 `--no_cacheblend`)。
  * **SGLang 和 RAGCache:** `LMCache` 仓库可能没有直接提供这些基线的实现。你可能需要参考它们各自的官方实现，并在相同的硬件和数据集上运行实验以进行公平比较。

#### **3. 复现微基准测试 (Microbenchmarks)**

这部分实验用于剖析 `CacheBlend` 核心组件的性能。

  * **KV 缓存融合:** 在 `LMCache` 仓库中查找与 KV 缓存相关的测试脚本。这些脚本通常位于 `tests` 或 `benchmarks` 目录下。运行这些脚本可以测量不同长度下 KV 缓存融合的延迟。
  * **选择性重计算:** 同样，在 `tests` 或 `benchmarks` 目录中寻找与重计算策略相关的脚本。通过运行这些脚本，你可以评估其在节省计算和内存方面的效果。

#### **4. 复现消融研究 (Ablation Study)**

消融研究旨在分析 `CacheBlend` 各个组成部分的重要性。

  * **方法:** 你需要修改实验脚本，以禁用或替换 `CacheBlend` 的特定模块。例如，你可以：
      * 禁用 KV 缓存融合，使用简单的拼接代替。
      * 禁用选择性重计算，总是执行完整的重计算。
  * **运行:** 修改脚本后，重新运行端到端 RAG 实验，并记录性能变化。将结果与完整的 `CacheBlend` 性能进行比较，以评估被禁用模块的贡献。

-----

### **具体实验执行流程**

#### **实验环境设置**

在当前 LMCache_retry 目录下，已包含完整的 CacheBlend 实验代码。具体执行步骤如下：

**1. 安装依赖**
```bash
# 安装核心依赖
pip install -r requirements/common.txt
pip install -r requirements/cuda.txt

# 安装 RAG 实验相关依赖
cd benchmarks/rag
pip install -r requirements.txt
```

**2. 配置模型和数据集**

需要准备以下文件：
- 模型：`mistralai/Mistral-7B-Instruct-v0.2` (通过 Hugging Face 自动下载)
- 数据集：下载 `musique_s.json` 到 `~/CacheBlend/inputs/` 目录
  - 数据集格式：包含 `ctxs`（文档）、`question`（问题）、`answers`（答案）字段
  - 参考: [CacheBlend musique_s.json](https://github.com/YaoJiayi/CacheBlend/blob/main/inputs/musique_s.json)

#### **端到端 RAG 性能实验详细执行**

**步骤 1：启动服务器**

1. **LMCache+CacheBlend 服务器:**
```bash
cd benchmarks/rag
export LMCACHE_CONFIG_FILE=example_blending.yaml
python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization 0.7 \
    --port 8000
```

2. **标准 vLLM 服务器 (基线对比):**
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
    --disable-log-requests \
    --port 8000
```

**步骤 2：运行 CacheBlend 实验**
```bash
cd benchmarks/rag
./launch_lmcache.sh
```
此脚本执行：
- 预计算阶段：计算能在指定存储大小内缓存的文档数量
- 基准测试：以 3.5 QPS 运行 RAG 任务，测量 TTFT 和吞吐量

**步骤 3：运行基线对比实验**
```bash
# 修改 launch_vllm.sh 中的 END_INDEX 为预计算输出的值
# 例如：END_INDEX=150 (根据预计算阶段输出调整)
./launch_vllm.sh
```

**关键配置参数:**
- `KV_STORAGE_SIZE=30GB`: KV 缓存存储大小
- `KV_CHUNK_SIZE=256`: 分块大小，需与 `example_blending.yaml` 中 `chunk_size` 一致
- `QPS=3.5`: 查询频率
- `--separator "[BLEND_SEP]"`: CacheBlend 专用分隔符

#### **微基准测试 (Microbenchmarks) 执行**

**1. KV 缓存融合测试:**
```bash
cd examples/blend_kv

# 离线测试
LMCACHE_CONFIG_FILE=example_blending.yaml \
LMCACHE_USE_EXPERIMENTAL=False \
python3 blend_kv.py

# 批处理测试
LMCACHE_CONFIG_FILE=example_blending.yaml \
LMCACHE_USE_EXPERIMENTAL=False \
python3 batched_kv.py
```

**2. 在线 KV 缓存测试:**
```bash
# 启动服务器
LMCACHE_CONFIG_FILE=example_blending.yaml \
LMCACHE_USE_EXPERIMENTAL=False \
CUDA_VISIBLE_DEVICES=0 \
python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --gpu-memory-utilization 0.8 \
    --port 8000

# 运行测试
python3 online_kv.py 8000
```

**3. 长文档 QA 基准测试:**
```bash
cd benchmarks/long-doc-qa
python3 long-doc-qa.py \
    --num-documents 10 \
    --document-length 20000 \
    --output-len 100 \
    --repeat-count 2 \
    --port 8000 \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

#### **消融研究 (Ablation Study) 执行**

**1. 禁用 KV 缓存融合:**
修改 `example_blending.yaml`:
```yaml
enable_blending: False  # 禁用融合功能
chunk_size: 256
local_device: "cpu"
```

**2. 修改缓存存储配置进行对比:**
```yaml
# 测试不同 chunk_size 的影响
chunk_size: 64   # 或 128, 512
enable_blending: True
local_device: "cpu"
```

**3. 重新运行实验:**
```bash
# 使用修改后的配置重新运行 RAG 实验
./launch_lmcache.sh
```

#### **性能指标收集**

实验结果将保存为 CSV 文件，包含以下关键指标：
- **TTFT (Time to First Token)**: 首字符生成时间
- **吞吐量**: 每秒处理的请求数
- **质量分数**: 基于 F1 和 Rouge-L 计算

输出文件示例：
- `musique_s_lmcache_qps_3.5.csv`: LMCache 结果
- `musique_s_vllm_qps_3.5.csv`: 标准 vLLM 结果

#### **实验验证检查清单**

- [ ] 确保 GPU 内存充足（推荐 > 16GB）
- [ ] 验证 `example_blending.yaml` 配置参数匹配
- [ ] 检查数据集路径 `~/CacheBlend/inputs/musique_s.json` 存在
- [ ] 确认服务器启动成功（端口 8000 可访问）
- [ ] 验证预计算阶段输出的 end_index 值
- [ ] 检查输出 CSV 文件生成

-----

### **注意事项与建议**

  * **仔细阅读 README:** `LMCache` 的 `README.md` 文件是复现实验最重要的信息源。
  * **配置一致性:** 确保 `KV_STORAGE_SIZE`、`KV_CHUNK_SIZE` 与配置文件中的参数匹配。
  * **硬件要求:** 至少需要一块 16GB+ VRAM 的 NVIDIA GPU。
  * **数据集准备:** 从 CacheBlend 官方仓库下载相应的数据集文件。
  * **环境变量:** 注意设置 `LMCACHE_CONFIG_FILE` 等环境变量。
  * **社区支持:** 遇到问题可在 LMCache GitHub 仓库提 Issue。

通过以上详细的执行流程，可以完整复现 CacheBlend 论文中的所有核心实验结果。