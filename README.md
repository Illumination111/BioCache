# BioCache

LIF 三维分层阵列存储 KV 的最小可行性研究工程。当前实现技术验证文档中的 **E0（拓扑与传播）和 E1（合成 KV 保持、探针和 Query 读取）**，并保留后续真实 VLM 实验所需的可复用基础模块。

## 安装

开发环境位于 `/data/user25101366/.conda/envs/biocache`，由 Conda 管理 Python，由环境内 pip 安装 PyTorch 和项目依赖。

```bash
conda create --prefix /data/user25101366/.conda/envs/biocache python=3.11 pip -y
conda activate /data/user25101366/.conda/envs/biocache
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints.txt -e '.[dev]'
```

PyTorch 使用 CUDA 12.8 wheel；开发机为 NVIDIA A100、驱动 580.178.04。安装方式依据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)。无需替换系统 CUDA toolkit。也可以在项目根目录运行 `conda env create -f environment.yml`。当前安装的精确依赖见 `requirements-lock.txt`，Conda 包见 `conda-explicit.txt`。

## 运行

从项目根目录执行；运行阶段无需联网或下载模型权重。输出目录必须不存在，防止覆盖已有结果。

```bash
pytest -q
python experiments/lif_kv_feasibility/run_propagation.py --output outputs/propagation
python experiments/lif_kv_feasibility/run_synthetic.py --output outputs/synthetic_memory
python experiments/lif_kv_feasibility/build_report.py
# 可选 GPU 实验，按实际空闲设备指定：
python experiments/lif_kv_feasibility/run_synthetic.py --device cuda:1 --output outputs/synthetic_cuda
```

默认合成实验在 CPU 上运行 3 个种子 × 4 个变体（LIF、EMA、断链、冻结 W）。GPU 版 PyTorch 同样支持 CPU 执行。配置中的 `steps`、数据规模及设备可修改；修改后的 resolved config 会随输出保存，不覆盖原预登记配置。

每次运行保存配置、运行版本、命令和 JSON 指标。E0 输出膜电位热图、发放 raster、首次发放时刻。E1 输出保持曲线、逐种子训练/验证误差、探针、三种读取位置、对照、梯度范数、checkpoint、状态与参数字节数、计时和验收报告。`outputs/` 默认不入 Git；初始运行的精简证据保存在 `experiments/lif_kv_feasibility/results/`。

## 实现范围

```text
src/biocache/
  config.py                  配置与拓扑约束
  data.py                    四类合成 KV 与教师 attention
  memory/layered_array.py    同步 LIF / EMA，状态生命周期
  memory/readout.py          带符号编码器、探针、latent KV 读取
  artifacts.py               实验记录
experiments/lif_kv_feasibility/
  configs/                   预登记传播与合成实验配置
  run_propagation.py          E0
  run_synthetic.py            E1 训练、评估、保存恢复、内存验证
tests/                       动力学、因果、层隔离、梯度与 CUDA 检查
```

输入按 token 顺序为 `[B,L,Hkv,Dh]`；状态 `[B,L,C,N]`。每层独立编码 K/V 与因果位置/时间，只向首列首个微步注入电流。Query 只参与读取，读取不修改状态，推理显式使用 `torch.no_grad()`。学生状态不保存历史 K/V，完整历史仅在实验教师对象中存在。

合成数据覆盖低秩重复背景、随机关联、短暂唯一事件和相似 Key / 不同 Value；训练和测试使用独立种子与序列。延迟为 0/16/64/256 事件，每事件恰好 S 个微步，包含零输入等待和干扰写入。读取器生成固定数量 latent K/V，不重建历史 token。末列是主要验收对象，首列和全列结果同时报告。

## 当前限制

这是可运行的研究原型，实验成功与否以 [实测报告](experiments/lif_kv_feasibility/report.md) 为准。代码可训练、传播正确、状态有界不代表末列已经有效存储任意历史 KV。

E2 真实 SmolVLM trace、E3 真实逐层蒸馏、E4 模型内替换未执行；模型 revision 字段刻意保留为 `REQUIRED_BEFORE_REAL_MODEL_RUN`，合成入口不会加载该模型。完整建立指南中的 toy Transformer、近期/保护精确缓存、路由、融合、真实模型适配、理论分析不在本轮实现范围，P0/P1/P2/P3 全框架验收均未宣称完成。

当前合成 runner 为单层，基础阵列支持多层并通过隔离测试。全列读取器参数更多；EMA 使用与 LIF 相同的两个 FP32 缓冲，其中第二个为冗余 EMA 副本，按实际字节统计。默认训练延迟最大 16，64/256 属于外推测试。没有端侧加速、能耗或真实视频问答质量结论。

详细数据流见 [architecture](docs/architecture.md)，设计边界见 [decisions](docs/decisions.md)。原始两份设计文档保留在 `docs/`。

初始实测：17 项测试通过，E0 通过；E1 非零延迟条件 3/24 达到预登记标准，整体尚未通过。1,000 与 10,000 事件持久状态均为 2,061 字节。训练改善与有界性不能替代存储质量验收，详见实测报告。
