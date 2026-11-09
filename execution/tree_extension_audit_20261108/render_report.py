"""Render the local audit; no network, model loading, or training."""
import hashlib
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = [
    "experiments/ac_seed42_v1_20261106/advantages.py",
    "experiments/prefix_protection_20261107/token_loss.py",
    "experiments/prefix_protection_20261107/runtime_mixin.py",
    "experiments/prefix_protection_20261107/native_sync.py",
    "experiments/prefix_protection_20261107/native_grpo_trainer.py",
    "execution/length_norm_pair_seed17_48_20261107/completed_evidence/comparison.json",
    "execution/length_norm_pair_seed17_48_20261107/result_summary.json",
    "execution/prefix_triplet_seed17_48_20261107/completed_evidence/comparison.json",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}


before = hashes()
cpu = json.loads((HERE / "cpu_verification.json").read_text())
cost = json.loads((HERE / "cost_projection.json").read_text())
assert cpu["passed"] and not cpu["gpu_used"] and cpu["no_policy_update"]


def linked(text):
    escaped = html.escape(text)
    return re.sub(r"https://[^\s<]+", lambda m: f'<a href="{m.group()}">{m.group()}</a>', escaped)


blocks = (HERE / "REPORT.txt").read_text().strip().split("\n\n")
paragraphs = []
for index, block in enumerate(blocks):
    lines = block.splitlines()
    if index == 0:
        continue
    if re.match(r"^[1-7]\. ", lines[0]):
        paragraphs.append(f"<h2>{html.escape(lines[0])}</h2>")
        lines = lines[1:]
    if lines:
        paragraphs.append('<p>' + linked("\n".join(lines)).replace("\n", "<br>") + '</p>')

table = """<table><thead><tr><th>建议短试臂</th><th>rollout / 信号</th><th>作用</th></tr></thead><tbody>
<tr><td>chain_cb</td><td>原链式 C-bounded</td><td>继续训练的控制</td></tr>
<tr><td>tree_vr</td><td>树状采样＋验答树信用</td><td>检验树状训练</td></tr>
<tr><td>tree_cb</td><td>同一树算法＋固定有界 PRM 项</td><td>检验 PRM 的增量</td></tr>
</tbody></table>"""
page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>树状 rollout 与 C-bounded · 审查与执行方案</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1e293b;font:16px/1.85 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}main{max-width:1000px;margin:40px auto;padding:44px 52px;background:white;border:1px solid #dce3eb;border-radius:14px}h1{font-size:29px;line-height:1.4;margin:0 0 12px}h2{font-size:21px;line-height:1.5;margin:36px 0 14px;border-top:1px solid #e2e8f0;padding-top:24px}p{margin:16px 0;overflow-wrap:anywhere}.state{color:#476582;font-size:14px}.callout{padding:18px 22px;background:#edf5ff;border-left:4px solid #3b70aa;border-radius:4px;margin:24px 0}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.fact{padding:16px;background:#f6f8fb;border:1px solid #e1e8f1;border-radius:8px}.fact b{display:block;font-size:24px;line-height:1.4}.fact span{font-size:13px;color:#526579}table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce3eb}th{background:#f3f6fa}a{color:#215e9d;text-decoration:none}a:hover{text-decoration:underline}footer{margin-top:32px;padding-top:20px;border-top:1px solid #e2e8f0;color:#5e7189;font-size:14px}@media(max-width:700px){main{margin:0;padding:26px 20px;border:0;border-radius:0}.facts{grid-template-columns:1fr}h1{font-size:25px}table{font-size:13px}th,td{padding:8px}}
</style><main><h1>树状 rollout 与 C-bounded</h1><div class="state">2026-11-08 · 原始论文、官方代码与历史吞吐核对</div>
<div class="callout"><b>建议：做一个有边界的树状信用扩展，保留现有主结果。</b><br>本轮完成文献/代码审查和 CPU 算例；树采样器、真实模型无更新联调与新训练均未执行。</div>
<div class="facts"><div class="fact"><b>16 / 16</b><span>CPU 公式边界检查；非真实模型验收</span></div><div class="fact"><b>0</b><span>新训练 / GPU 作业</span></div><div class="fact"><b>1.397 h / M</b><span>历史链式作业墙钟参照；非树状实测</span></div></div>
""" + table + "\n" + "\n".join(paragraphs) + """<footer>证据文件：<a href="sources.json">来源与版本</a> · <a href="plan.json">建议实验合同</a> · <a href="cpu_verification.json">CPU 核验</a> · <a href="cost_projection.json">成本重算</a> · <a href="REPORT.txt">纯文本报告</a></footer></main></html>"""
(HERE / "review.html").write_text(page)
after = hashes()
assert after == before, "Historical inputs changed during local report rendering"
(HERE / "historical_inputs.json").write_text(json.dumps(dict(before=before, after=after, unchanged=True), indent=2) + "\n")
output_names = ["REPORT.txt", "review.html", "sources.json", "plan.json", "cpu_credit_contract.py",
                "cpu_verification.json", "cost_projection.json", "historical_inputs.json", "render_report.py"]
receipt = dict(scope="Read-only source audit and independent local CPU algebra/reference; no remote execution.",
               historical_inputs_unchanged=True, gpu_jobs_started=0, training_updates=0,
               files={name: dict(bytes=(HERE / name).stat().st_size, sha256=hashlib.sha256((HERE / name).read_bytes()).hexdigest()) for name in output_names})
(HERE / "delivery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(report=str(HERE / "review.html"), cpu_checks=cpu["count"],
                     receipt_files=len(output_names), historical_inputs_verified=len(before),
                     hours_per_million=cost["measured_chain_hours_per_million"]), indent=2))
