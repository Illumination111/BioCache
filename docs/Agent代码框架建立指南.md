# 流式 VLM 神经元 KV-Cache 压缩：Agent 代码框架建立指南

> 依据：同目录《工作设计.pdf》，武博文，2026-10-31，共 27 页。
> 本文用于指导代码 agent 建立可运行、可训练、可评估的研究框架。当前交付是实现指南，尚未实现代码或验证研究效果。
> 拓扑修订：依据用户进一步明确的三维分层阵列定义；该定义优先于本文先前的固定空间槽 / 局部连接建议。
> 标记约定：【原文】表示 PDF 明确提出的设计；【建议】表示为落地而补充的工程选择；【待研究】表示尚需实验判断的假设。本文给出的参数均为冒烟测试起点，不是论文结论。

## 1. 给执行 Agent 的任务

在 `Paper_NIPS_27/neural_kv/` 中建立 Python/PyTorch 研究工程，实现面向流式视觉语言模型的双记忆 KV 压缩框架：

1. 对每层历史视觉 KV 做语义编码与降维，写入有界的 LIF 神经元状态。
2. 使用路由器，将需要精确检索的信息保存在精确 KV 分支。
3. 让当前层 Query 读取压缩状态，与精确 KV 的读取结果融合，继续原模型推理。
4. 冻结预训练 VLM，先训练新增模块，通过注意力输出蒸馏验证可行性。
5. 提供合成数据闭环、真实模型适配接口、基线、资源统计和研究消融入口。

先建立 CPU 可运行的最小闭环，再接入一个实际 VLM。不得把完整历史 KV 隐藏保存在对象、日志、重建缓存或 GPU tensor 列表中，并宣称实现有界压缩。无需在首版开发动态拓扑、自定义 CUDA 内核或修改视觉编码器。

## 2. 原文设计及实现边界

| 来源 | 明确内容 | 对代码框架的要求 |
|---|---|---|
| 第 3-7 页 | 流式视觉历史增长与端侧资源矛盾，LIF 具有时间积分、泄漏和事件更新能力 | 持续输入、有限状态、可测量的内存与更新开销 |
| 第 14-16 页 | LIF 阵列与精确 KV 双精度缓存；语义感知路由；Query 读取 | 路由、压缩写入、精确缓存、读取融合分别建模块 |
| 第 17 页图 | Prefill → 分层 KV → Semantic Encoder → 降维 → LIF；读取时结合近期 KV 解码 | 压缩对象是各 Transformer 层 KV；保留近期显式缓存路径 |
| 第 18 页 | 规整阵列与动态演化阵列均存在取舍 | 首版固定布局；动态拓扑作为独立后续选项 |
| 第 21-24 页 | LIF 时间动力学、空间耦合、区域聚合 | 多时间尺度与空间结构可配置，显式保存位置元数据 |
| 第 25 页 | 以未来 Query 的注意力输出误差定义有损压缩 | 主要目标为输出蒸馏，不强制可逆恢复历史 KV |
| 第 26-27 页 | 条件方差、碰撞误差、低秩误差分析 | 提供分析实验，明确各公式成立条件 |

【待研究】PDF 没有确定具体 VLM、数据集、训练规模、端侧硬件、读取网络、路由监督或可接受精度损失。不得将这些选择写成原文结论，也不得将文献页中的效果当成本方法已实现的效果。

【用户明确】首版使用三维分层 LIF 阵列，每个 Transformer 层对应一片二维阵列；片内相邻列之间全连接，当前片间无突触连接。采用软重置 LIF、启发式路由、直接 Query-conditioned readout。模型、数据、设备由配置指定；未提供真实权重时完成合成验证并如实报告真实集成未验证。

## 3. 总体数据流与因果时序

```text
视频帧 / 文本 → 原 VLM 的视觉编码与 prefill → 每层 Q、K、V 及元数据
                                                   │
                              当前可见精确 KV + 当前可见 LIF 状态
                                                   ↓
                           精确注意力 + Query-conditioned LIF 读取
                                                   ↓
                                      融合 → 原层输出投影
                                                   │
                         已处理 token 进入近期精确窗口
                                                   ↓
                        从窗口迁出的 token → 路由器
                                      ├─ 精确保留 → 受预算约束的保护缓存
                                      └─ 可压缩 → 编码 / 降维 → LIF 更新
```

【建议】近期窗口、保护缓存、LIF 压缩分支在首版使用互斥归属：token 从窗口迁出后只进入一个长期分支，避免重复计数。先用此策略得到可解释基线；同时写入两种长期记忆作为后续消融，必须解决重复贡献问题。

必须定义并实现 `prefill` 与 `decode` 的等价因果规则：

- 单 token 参考实现：当前注意力包含自己及合法前缀的显式 KV；读取旧 LIF 状态；完成当前输出后提交缓存迁移与状态更新。
- 新输入块不得提前整体写入 LIF，防止块前部 Query 读取未来 token。
- 首版允许逐 token 参考执行；优化块处理时必须与参考实现比较，不能只检查张量形状。
- 避免跨层误写：每层只消费本层 KV；每 token、每层仅更新一次。
- `read()` 默认只读；时间推进与写入由独立 `advance/update` 完成。若按时间间隔衰减，即使没有写入也需处理时间推进。
- 当前问句可影响当前及之后的路由，历史压缩不得使用当时尚不可见的未来问题或答案。

## 4. 推荐目录

```text
neural_kv/
├── pyproject.toml
├── README.md
├── configs/
│   ├── synthetic.yaml
│   ├── distill.yaml
│   ├── streaming_eval.yaml
│   └── ablations/
├── src/neural_kv/
│   ├── config.py                 # 配置加载、约束与版本
│   ├── types.py                  # 张量与元数据契约
│   ├── adapters/
│   │   ├── base.py               # VLM 适配协议
│   │   ├── toy_transformer.py    # 小型因果模型，真实走注意力路径
│   │   └── hf_vlm.py             # 后续选择一个模型完成适配
│   ├── memory/
│   │   ├── exact_cache.py        # 近期窗口与保护缓存
│   │   ├── semantic_encoder.py   # KV、位置、时间信息编码
│   │   ├── lif.py                # 动力学与代理梯度
│   │   ├── layered_array.py      # 三维分层阵列；片内相邻列全连接
│   │   ├── readout.py            # Query 读取状态
│   │   ├── router.py             # 规则版 / 可学习版
│   │   ├── fusion.py             # 双分支融合
│   │   └── manager.py            # 每层状态与迁移事务
│   ├── attention/hybrid.py       # 混合注意力参考实现
│   ├── data/
│   │   ├── synthetic.py          # 稳定背景、瞬时事件、空间回溯
│   │   ├── stream.py             # 单调时间输入与会话隔离
│   │   └── traces.py             # 教师 QKV / attention output
│   ├── training/
│   │   ├── losses.py
│   │   └── distill.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── baselines.py
│   └── analysis/
│       ├── collision.py
│       └── low_rank.py
├── scripts/
│   ├── smoke.py
│   ├── collect_traces.py
│   ├── train_distill.py
│   ├── eval_streaming.py
│   └── analyze_memory.py
├── tests/
└── outputs/                     # 默认 gitignore
```

避免为未实现的研究选项生成大量空文件。接口允许尚未实现，但必须明确抛出 `NotImplementedError`；最终交付的最小闭环不得依赖此类接口。

## 5. 张量契约与生命周期

【建议】首版采用以下显式形状，禁止依赖隐式 broadcast：

| 名称 | 形状 / 内容 |
|---|---|
| `Q` | `[B, Hq, Nq, Dh]` |
| `K, V` | `[B, Hkv, Nk, Dh]` |
| `TokenMeta` | 每 token 的 `stream_id, token_id, absolute_position, timestamp, modality, frame_id, spatial_coord, valid_mask` |
| `LIFState.u, .spikes` | 整体 `[B, L, C, N]`；每层二维片 `[B, C, N]`，C 列、每列 N 个标量 LIF |
| `W_forward` | `[L, C-1, N, N]`；每片各相邻列的稠密突触矩阵 |
| `LIFState.last_timestamp` | 时间衰减所需的每样本 / 层时间信息 |
| `RouteDecision` | 精确保留 / 压缩 mask、风险分数及原因；互斥且覆盖有效迁出 token |
| `ReadResult` | `[B, Hq, Nq, Dh]`；附带状态是否有效等信息 |
| 精确缓存 | 本层 K/V 与逐条位置、时间、mask；近期和保护区总量受限 |

GQA 要求 `Hq % Hkv == 0`，显式实现 Query head 到 KV head 的对应关系；不要为方便永久复制 KV 到 `Hq` 份。每层编码器将本层所有 KV heads 联合编码到首列 N 个神经元；读取器再输出各 Query head 的结果。head 维度不是阵列新增的几何轴。

API 至少包括：

```python
class MemoryManager:
    def reset(self, stream_ids): ...
    def read(self, layer_id, q, query_meta): ...
    def commit(self, layer_id, k, v, meta): ...
    def detach_state(self): ...
    def export_state(self): ...
    def import_state(self, state): ...
    def memory_stats(self): ...
```

要求：

- `reset` 清除精确缓存、LIF、计数器、时间戳和路由历史；不同视频、样本、会话不得串扰。
- 首版允许只支持推理 batch size 1，但应显式校验；未支持 beam search、批次重排时拒绝执行。
- 推理使用 `no_grad`；训练采用截断 BPTT，在片段边界 detach，避免计算图随视频无限增长。
- 序列化包含配置版本、模型标识、层映射、dtype 与位置约定；恢复后下一步结果应一致。
- 变分辨率帧的位置由编码器处理，C、N 固定，不能随帧数增长；神经元行号不默认等同于 patch 坐标。

## 6. 核心模块实现约束

### 6.1 三维阵列拓扑与 KV 编码

【用户明确】整个阵列以 `(模型层 l, 列 c, 行 n)` 为三维坐标。第 l 个 Transformer 层生成的 KV 只保存在第 l 片二维 LIF 结构中。每片的每列包含 N 个神经元，每个神经元连接下一列全部 N 个神经元，因此每对相邻列有 N² 条突触。

```text
Transformer 层 0 的 KV → 片 0：列 0 ==全连接==> 列 1 ==> … ==> 列 C-1
Transformer 层 1 的 KV → 片 1：列 0 ==全连接==> 列 1 ==> … ==> 列 C-1
…                         各片之间当前没有突触连接
Transformer 层 L-1 KV → 片 L-1：列 0 ==> 列 1 ==> … ==> 列 C-1
```

这里“全连接”指相邻列之间的稠密前向二部连接，不是片内任意两个神经元全互连。首版无同列横向连接、反向连接、跳列连接或跨片连接。时间状态保持是单神经元动力学，不视为跨神经元突触。

【建议】每层独立编码 `concat(flatten_heads(K), flatten_heads(V), position_embedding, time_embedding)`，映射到首列 N 维输入电流。位置来自真实视觉 token / merge 后 patch 映射；不强制将 patch 坐标映射为神经元行号。文本默认留在精确分支。同帧多个 token 默认按因果 token 顺序逐次输入首列。

【后续可能研究】在不同 Transformer 层对应的二维片之间增加全连接突触。首版配置 `cross_layer_connections: false`，不分配跨层参数。未来必须另行定义层方向、更新同步和因果掩码，并评估参数 / 计算增长及层间信息污染；不得悄悄开启。验证文档仅验证片内连接。

### 6.2 LIF 动力学与列传播

每个神经元的膜电位是标量。对本层一次输入事件，拆分 S 个内部微步；仅微步 0 向首列注入当前 KV 编码，其余微步外部输入为零。每微步使用旧状态同步更新全部列：

```text
I[l,0] = encoder_l(k,v,meta) if microstep == 0 else 0
I[l,c] = W_forward[l,c-1] @ spikes_old[l,c-1], c > 0
u_pre[l,c] = exp(-delta_micro / tau[l,c]) * u_old[l,c] + I[l,c]
spikes_new[l,c] = H(u_pre[l,c] - threshold[l,c])
u_new[l,c] = u_pre[l,c] - threshold[l,c] * spikes_new[l,c]
```

因此孤立输入最早在首个微步激活列 0，在第 C 个微步影响列 C-1（从 1 开始计数）。设 `S >= C` 可在一次事件内观测完整传播，但不能保证信号必然达到末列；阈值、权重、稀疏性和饱和均需验证。禁止原地从左到右更新而使同一微步跨越全部列。

- 采用当步软重置，不再重复减前一步 reset 项。
- `tau`、阈值为正；多时间尺度按列 / 行配置。微步时间与 token / 帧时间不同，必须分别记录。改变 S 时保持总时间或注明总时间改变。
- 不默认将带符号 KV 截断为正值。优先可训练电流投影，另比较成对正负通道编码；后者占用真实神经元容量。
- 保留所有列的膜电位与当步脉冲，不保留无界 spike 历史。
- 训练使用 surrogate gradient；推理前向仍为硬脉冲，二者动力学一致。
- 无脉冲 EMA、无列间连接、不同 C/S/阈值为必要消融。
- 密集 PyTorch 实现不因发放稀疏而自动节能；记录内部微步开销。

### 6.3 路由与缓存预算

【原文】重要性、数字 / 坐标 / 突发事件风险、当前状态表达能力共同决定路由。

【建议】先实现规则版，输入仅包含当前可见信息：模态、近期性、文本风险标签、相邻帧特征变化、由历史状态预测当前特征的残差。视觉数字风险需额外可用信号，不可假设原始 KV 自动提供 OCR 标签。

规则输出 `risk_score` 与可解释原因。注意力重要性如需显式计算，其额外成本必须计入；不得使用未来 Query 的注意力评分。

精确预算 `C_exact = C_recent + C_protected` 固定。近期窗口迁出后：低风险项写入 LIF，高风险项进入保护缓存。保护缓存满时，按显式优先级淘汰；淘汰项可再压缩并记日志。高风险 token 无法永远保留，必须统计强制淘汰率。文本超预算时也必须采用明确策略，不能默默突破容量。

可学习路由后续再加入；须说明离散选择的训练方法，如软门控或 straight-through estimator，不能只用不可导 top-k 后宣称端到端训练。

### 6.4 Query 读取与融合

【待研究】第 15-16 页保留逆运算与直接读取的可行性讨论；第 25 页已明确不要求恢复完整 KV。

【建议】首版从 `[u, spikes]` 从 C×N 状态生成固定数量 M 个 latent keys/values，投影到对应 head 维度，由当前 Q 做 attention 得到 `o_lif`。这些是状态读取槽位，不是恢复出的原始历史 token。

精确分支得到 `o_exact`，门控融合：

```text
g = sigmoid(gate(q, state_summary, cache_summary))
o = g * o_exact + (1 - g) * o_lif
```

门控融合是需要训练的近似，两个独立 softmax 的平均不等价于对完整历史 KV 做一次 softmax。后续可比较联合 latent/exact softmax，仍应明确 latent 槽与原 token 的归一化质量并非天然一致。

必须定义空分支：无 LIF 历史时只使用精确分支；无精确项时只使用有效 LIF 分支；全空返回明确零值或合法初始值，避免全 mask softmax 产生 NaN。混合结果进入原模型输出投影之前，张量形状与原注意力输出一致。

### 6.5 模型适配与位置编码

先完成 toy causal transformer；再选定一个支持视频 / 多图流输入的实际模型，固定模型 ID、revision、依赖版本与视觉 token 映射方式。

- 通过显式 attention wrapper 替换读取路径；只用 forward hook 收集 KV 并不能实现真正的压缩推理。
- 明确 K 是 RoPE 前还是 RoPE 后。首版可让精确缓存保存原模型实际使用的旋转后 K，并将该约定写入配置。
- 精确项保留原始 absolute position，不因物理压缩重新编号；兼容模型的多维位置编码时由适配层负责。
- latent key 的位置处理由读取器定义，不要对已经旋转的 K 重复应用 RoPE。
- 禁用压缩时与原模型输出对齐；启用后确认原完整 `past_key_values` 已释放，不存在双份缓存。
- 区分视觉 prefill、文本 prefill 与 decode。PDF 首要任务是 KV 压缩；不自动声称减少视觉编码成本。

## 7. 训练闭环

【原文，第 25 页】目标是最小化完整历史注意力和压缩读取的输出差异：

```text
Y = Attention(Q, K_history, V_history)
Z_t = F_theta(Z_{t-1}, K_t, V_t)
Y_hat = G_phi(Q, Z_t, exact_cache)
L_attn = mean(||Y - Y_hat||_2^2)
```

双分支版本是对原文压缩目标的工程扩展，教师与学生必须对应相同可见历史、位置和 Query。

推荐分三步：

1. **离线 QKV 实验**：以合成关联检索或真实教师 trace 训练 encoder、LIF、readout、fusion，验证注意力近似误差是否下降。
2. **模型内集成**：冻结 VLM 主干，仅训练新增模块，加入选定层 attention output 蒸馏。离线教师 Q 与在线学生 Q 可能分布不同，需要在此阶段检验误差累积。
3. **可选任务微调**：加入 logits KL、答案交叉熵；可学习路由与活动率约束逐项加入，保持可消融。

新增正则项需单独记录权重和量纲。`budget_loss` 不替代硬缓存容量约束；脉冲活动率也不是能耗。教师完整 KV 仅允许在明确标记的训练 / 评估参考路径存在，不计入学生部署缓存；训练峰值内存应另报。

记录训练与验证 loss、各参数梯度范数、膜电位范围、发放率、门控分布、路由比例。按视频 / 会话划分训练验证集合，避免将同一视频相邻片段拆到不同集合形成泄漏。

## 8. 配置与命令契约

以下是建议默认值，agent 应实现并验证，不代表当前命令已存在：

```yaml
seed: 42
model:
  backend: toy
  layers: 2
  query_heads: 4
  kv_heads: 2
  head_dim: 16
memory:
  recent_tokens: 64
  protected_tokens: 64
  columns: 4
  neurons_per_column: 64
  microsteps_per_token: 4
  topology: adjacent_columns_dense
  cross_layer_connections: false
  reset: soft
  time_unit: token_event
  tau_init: [16.0, 64.0, 256.0]
  threshold_init: 1.0
router:
  type: heuristic
readout:
  type: latent_attention
fusion:
  type: learned_gate
training:
  freeze_backbone: true
  tbptt_steps: 32
  loss: attention_mse
runtime:
  device: cpu
  dtype: float32
```

多时间尺度参数需明确分配到各列 / 行；校验维度、容量、head 整除、时间单调、dtype 与设备兼容。真实模型的层数与维度从模型读取，禁止套用 toy 配置硬编码。

最终 README 应提供以下可执行入口，并记录实际运行结果：

```bash
python -m pip install -e '.[dev]'
python scripts/smoke.py --config configs/synthetic.yaml
pytest -q
python scripts/train_distill.py --config configs/distill.yaml
python scripts/eval_streaming.py --config configs/streaming_eval.yaml
python scripts/analyze_memory.py --config configs/synthetic.yaml
```

依赖缺失、无网络或无权重时如实报告；不得用随机模型输出替代真实 VLM 质量结果。框架 CPU 冒烟不得依赖外网下载。

## 9. 基线、评估和理论分析

### 9.1 最小对照

- 完整 KV：质量参考，缓存随历史增长。
- 固定滑动窗口：不带长期记忆。
- 同预算精确缓存：近期窗口加规则保留，不带 LIF。
- 双记忆 LIF：本方法。
- 双记忆 EMA：相同状态容量、相同读写器，去掉阈值发放和 reset，检验 LIF 的独立贡献。

H₂O、KIVI、Palu、MuKV、VLMCache、Interdomain Attention、HOLA 在 PDF 中属于相关工作。首版不要求全部复现。实现简化版时必须标为简化对照，不能直接冒用论文方法名；若后续接入，应核对官方实现和适配条件。

### 9.2 任务与指标

合成任务至少覆盖稳定背景、瞬时事件、延迟关联检索、空间位置区分和高风险信息保护；改变回溯距离、背景重复率、事件稀疏率和流长度。随机 KV 仅能检验数值接口，不能替代这些任务。

真实任务由后续选定的数据集提供流式视频问答或历史回溯；报告具体采帧、分辨率、视觉 token 数、问句到达时刻和答案指标。不同方法使用相同输入、生成参数、种子与设备。

| 维度 | 必须输出 |
|---|---|
| 近似质量 | attention MSE、相对 L2；可选 logits KL |
| 任务质量 | QA 指标、事件 / 数字 / 空间细节检索表现、回溯距离曲线 |
| 持久内存 | 精确 KV、LIF、路由元数据、时间位置、其他运行状态的字节数 |
| 实际内存 | 峰值设备分配与运行态；教师与学生分别报告 |
| 延迟 | prefill、decode、压缩写入、读取融合耗时及 p50/p95 |
| 吞吐 | 帧吞吐与 token 吞吐，固定输入条件 |
| 行为 | 强制淘汰率、保护命中率、发放率、路由比例 |

标准缓存估算为 `2 * B * L * T * Hkv * Dh * bytes_per_element`。LIF 至少计算 `[B,L,C,N]` 的 `u` 与 `spikes` 实际存储，再计入额外状态。列间参数量为 `L*(C-1)*N*N`，计入新增模块参数；同时计入编码器、读取器和训练优化器开销。新增模块参数与临时激活另列；压缩比同时提供持久缓存口径与端到端内存口径。同预算对照应按字节匹配，而非仅匹配 token 数。

GPU 计时需 warmup 与同步；CPU 分别报告。端侧硬件未实测前，只能称其为端侧部署目标，不能给出端侧加速或功耗结论。

### 9.3 理论实验的适用范围

第 26 页：固定编码器、平方误差下，最优读取器是 `E[Y | Q,Z]`，剩余风险为条件方差的期望。`1/4 * ||y1-y2||²` 对应同一可见 `(Q,Z)` 下两个等概率历史的碰撞情形；非等概率为 `p(1-p)||y1-y2||²`。实现受控碰撞样例，不能把有限样本近邻误差宣称为已证明的一般下界。双记忆分析的条件还须包含精确缓存，不能只看 Z。

第 27 页：截断 SVD 的尾部平方奇异值和，是秩约束下 Value 矩阵 Frobenius 重构误差的最优值。实现谱衰减和截断重构检查；该结论不是非线性 LIF 的通用注意力误差下界，也不能单凭谱衰减保证实际任务损失可接受。

## 10. 分阶段执行及验收

### P0：契约与 CPU 闭环

建立包、配置、toy 模型、合成流、分层状态、有限精确缓存、LIF、直接读取和融合。`smoke.py` 实际完成流输入 → 写入 → 延迟 Query → 读取 → 输出指标。不能只有 dataclass 和空函数。

验收：CPU 无外网运行，张量形状合法、无 NaN、状态能 reset；长流测试中缓存占用在容量填满后保持有界。

### P1：蒸馏可训练

生成具有可学习结构的教师 QKV / attention trace，完成优化、保存和恢复。使用固定种子与独立验证集比较初始误差和训练后误差。训练改善阈值应在实验配置中预先声明；未改善时分析门控退化、遗忘和梯度问题，不伪造结果。

验收：新增模块有有限且非零的有效梯度；主干冻结；验证集输出误差有可复核改善；checkpoint 恢复后输出一致。

### P2：一个真实 VLM 适配

先通过禁用压缩的原模型对齐，再启用逐层混合注意力。验证流式输入语义、位置编码、GQA 和完整缓存释放。以小视频完成一次推理 / 蒸馏验证。

验收：记录模型与依赖版本、真实输入输出、资源数据和已知限制。无权重或硬件时将此阶段标记未完成，并交付可用适配接口与明确阻塞条件。

### P3：公平基线和消融

运行窗口、同预算精确缓存、LIF 与 EMA；再比较路由、列间连接、多时间尺度、状态容量、融合方式。输出 JSON/CSV 指标和曲线，保留实际配置与种子。

验收：能在相同输入和资源口径下复现实验，明确何时精度下降、何时额外读写成本抵消收益。

## 11. 必要测试

这些测试针对研究实现中的实质风险，优先于与代码结构重复的测试：

1. **拓扑与传播**：每片仅存在相邻列稠密连接；单片输入不改变其他片；末列不能早于第 C 个微步响应孤立输入；同步更新与参考循环一致。
2. **动力学**：无输入衰减解析值；阈值发放；软重置；时间间隔；训练与推理前向一致。
3. **因果性**：修改未来 token，不能改变已生成输出；块输入与逐 token 参考时序一致。
4. **路由与容量**：有效迁出项恰有一个目的地；缓存满后执行明确淘汰；高风险项过量时仍受预算约束。
5. **有限内存**：例如输入 1,000 与 10,000 个事件，饱和后持久状态 tensor 字节数一致；排查 Python 容器和计算图增长。
6. **会话隔离**：reset 后等价全新实例；不同视频不能共享状态。
7. **读取边界**：空精确 / 空 LIF、padding、混合 dtype 与 GQA 映射；所有输出有限。
8. **模型对齐**：禁用压缩、精确容量足够时，与标准注意力在约定容差内一致；有非连续原始位置时仍正确。
9. **训练和恢复**：关键模块获得梯度；主干无梯度；导出恢复后下一步状态和输出一致。
10. **理论工具**：SVD 尾能量等于截断重构误差；受控两历史碰撞符合给定概率公式。

## 12. Agent 最终交付清单

- 可安装的工程与实际可执行的 CPU 示例。
- README：安装、数据格式、配置、训练、评估、模型适配限制。
- `docs/architecture.md`：真实实现的数据流、形状、更新时序、位置约定和内存预算。
- `docs/decisions.md`：区分 PDF 原始设计、工程默认选择、仍待验证的假设。
- `outputs/<run_id>/`：resolved config、版本、种子、命令、指标、必要日志。
- 测试结果及真实执行的命令；没有运行的验证单独列出。
- 阶段完成状态：P0/P1/P2/P3；未完成项与具体原因。

执行顺序为 P0 → P1 → P2 → P3。不要因模型或硬件暂不可用而停下独立的 CPU 框架建设；也不要把完成 CPU 框架描述为完成真实 VLM 的端侧有效性验证。
