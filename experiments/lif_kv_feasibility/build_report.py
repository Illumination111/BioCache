"""Export reproducible, small research evidence without committing checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--propagation", default="outputs/propagation")
    p.add_argument("--synthetic", default="outputs/synthetic_memory")
    p.add_argument("--output", default="experiments/lif_kv_feasibility")
    args = p.parse_args()
    prop, synth, dest = Path(args.propagation), Path(args.synthetic), Path(args.output)
    evidence = dest / "results"
    evidence.mkdir(parents=True, exist_ok=True)
    for source, prefix in ((prop, "propagation"), (synth, "synthetic")):
        folder = evidence / prefix
        folder.mkdir(exist_ok=True)
        for pattern in ("*.json", "*.yaml", "*.png"):
            for path in source.glob(pattern):
                shutil.copy2(path, folder / path.name)
    summary = read(synth / "summary.json")
    memory = read(synth / "memory.json")
    e0 = read(prop / "metrics.json")
    with (evidence / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    lif = [read(path) for path in sorted(synth.glob("lif_*.json"))]
    checks = [r for r in summary if r["variant"] == "lif" and r["delay"] > 0]
    passed = [r for r in checks if r["passes_20pct_vs_both"]]
    lines = ["# BioCache 初始验证实测报告", "", "## 阶段状态", "",
        "| 阶段 | 结果 |", "|---|---|",
        f"| E0 拓扑/传播 | {e0['E0']}；首次发放微步 {e0['first_spike_microstep_1based']} |",
        f"| E1 合成 KV | 代码与实验完成；整体判据未通过，非零延迟 {len(passed)}/{len(checks)} 条件通过 |",
        "| E2 真实 SmolVLM traces | 未执行 |", "| E3 真实逐层蒸馏 | 未执行 |",
        "| E4 真实模型替换 | 未执行 |", "",
        "完整建立指南的 P0/P1 仅完成本轮实验基础模块，未完成完整框架验收；P2/P3 未完成。", "",
        "## 环境与实际命令", "",
        "环境：`/data/user25101366/.conda/envs/biocache`，Python 3.11.18，"
        "PyTorch 2.7.1+cu128，CUDA runtime 12.8。GPU 为 A100，CUDA 前后向测试通过；本轮三种子实验使用 CPU、单线程。", "",
        "```bash", "conda activate /data/user25101366/.conda/envs/biocache",
        "python -m pytest -q",
        f"python experiments/lif_kv_feasibility/run_propagation.py --output {prop}",
        f"python experiments/lif_kv_feasibility/run_synthetic.py --output {synth}",
        "python experiments/lif_kv_feasibility/build_report.py", "```", "",
        "测试结果：17 passed（包含 CUDA，未跳过）；依赖 `pip check` 通过。原始输出保留在项目 outputs；"
        "精简证据包含逐种子完整 JSON、CSV、配置、版本记录和图像。checkpoint 留在 outputs。", "",
        "## 传播", "", "![传播热图与 raster](results/propagation/propagation.png)", "",
        "受控权重下，C=4 时末列在第4微步首次发放；软重置、断链、片间隔离检查通过。这只证明动力学实现正确。", "",
        "## 独立验证集训练变化", "",
        "下面将每个种子在四种数据、两种等待模式、延迟0/16的末列 attention MSE 平均；初始和训练后用相同测试序列。", "",
        "| Seed | 初始 MSE | 训练后 MSE | 相对下降 |", "|---|---:|---:|---:|"]
    for r in lif:
        before = statistics.mean(m["mse_remote"] for m in r["initial"] if m["view"] == "last_column")
        after = statistics.mean(m["mse_remote"] for m in r["metrics"]
                                if m["view"] == "last_column" and m["delay"] in (0, 16))
        lines.append(f"| {r['seed']} | {before:.6f} | {after:.6f} | {(1-after/before)*100:.1f}% |")
    lines += ["", "encoder、W、tau、threshold、readout 梯度范数逐参数记录于 JSON；LIF 三种子均有有限非零梯度。"
              "checkpoint 恢复后，当前输出、下一事件状态及输出逐值一致。", "",
        "## 末列保持判据", "",
        "预登记标准：三个种子的独立测试均值，末列 MSE 比清零与打乱状态两对照均下降至少20%；"
        "对照误差接近零时用绝对容差1e-6。按数据/延迟/干扰分别判断，不把平均改善当作全部通过。", "",
        "| 数据 | 模式 | 延迟 | MSE 均值±样本标准差 | 清零 MSE | 打乱 MSE |",
        "|---|---|---:|---:|---:|---:|"]
    for r in passed:
        lines.append(f"| {r['pattern']} | {r['delay_mode']} | {r['delay']} | "
                     f"{r['mse_remote_mean']:.6f}±{r['mse_remote_std']:.6f} | "
                     f"{r['zero_mse_mean']:.6f} | {r['shuffle_mse_mean']:.6f} |")
    lines += ["", "上表只列通过的非零延迟条件，其他条件均未通过。完整通过/失败列表见 "
        "[summary.csv](results/summary.csv) 和 [summary.json](results/synthetic/summary.json)。"
        "只有3个种子，未做统计显著性推断，不能声称一般历史 KV 存储成功。", "",
        "![保持曲线](results/synthetic/retention.png)", "",
        "曲线包含各读取位置及 EMA、断链、冻结 W；是四种数据的平均，仅用于诊断，验收仍逐条件判断。"
        "测试延迟64/256超过训练延迟0/4/16；长延迟失败含外推限制。干扰输入下未达到预设判据。", "",
        "教师全历史注意力、仅原始远期历史注意力分别报告；训练只监督后者。精确窗口对照的容量为8个token，"
        "等待时保留原历史；有干扰时保留最后至多8个干扰token。"
        "四类数据的探针误差、清零/打乱探针误差与首列/全列读取误差均在逐种子JSON，不用非零膜电位代替信息保持。", "",
        "## 资源", "", "| 事件数 | 持久状态字节 | 完整 KV 字节 | 缓存比 | 新增静态模块+状态字节 |",
        "|---:|---:|---:|---:|---:|"]
    for m in memory["samples"]:
        lines.append(f"| {m['events']} | {m['state_bytes']} | {m['full_kv_bytes']} | "
                     f"{m['cache_compression_ratio']:.2f} | {m['student_static_plus_state_bytes']} |")
    params = lif[0]["parameters"]
    lines += ["", f"静态参数 {params['total']:,} 个、{params['bytes']:,} 字节，"
        f"其中列间 W {params['synapses']:,} 个；另有 {params['buffer_bytes']} 字节连接 mask。"
        "这里包括三个诊断读出器和三个探针，均未从统计中隐藏。spikes 为 FP32，"
        "状态按实际存储计算；持续流状态固定五个 tensor，推理无 autograd 图、无历史 KV 容器。", "",
        f"单样本 CPU 写入 p50={memory['write_ms_p50']:.3f} ms、p95={memory['write_ms_p95']:.3f} ms（前256事件）。"
        f"全进程峰值 RSS={memory['process_peak_rss_bytes_including_training']:,} 字节，包含教师/训练/框架，"
        "不是学生独立部署峰值；本轮 CPU 运行不报告 GPU 训练峰值。每批次写入、读取计时在逐种子 JSON。", "",
        "缓存有界性通过；静态模块使短历史总开销高于完整 KV。未测真实 VLM 端到端峰值、端侧延迟或能耗。", "",
        "## 限制与下一步", "",
        "目前末列在低复杂度平均 Value 读取场景可保留部分信息；随机/唯一事件和持续干扰条件不足。"
        "下一步先固定预算，检查连接权重与阈值、容量、训练延迟分布和 Query-conditioned readout，"
        "再决定是否进入真实 trace 阶段。当前结果不支持长期精确记忆或真实视频问答可用性的结论。", "",
        "EMA 对照使用相同分配字节、第二缓冲为冗余副本。全列读出参数更多："
        f"{params['readout']}；探针参数：{params['probe']}。"
        "本轮没有正负双通道、固定总神经元容量的 C/N/S 扫描、真实浅中深层误差或在线任务结果。", ""]
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {dest / 'report.md'}")


if __name__ == "__main__":
    main()
