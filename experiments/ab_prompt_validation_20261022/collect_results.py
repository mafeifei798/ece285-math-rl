"""One-shot result collection for this existing inference job; never launches GPU work."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import fcntl
import hashlib
import html
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/ab_prompt_validation_20261022'
REMOTE = '/path/to/ece285_math_rl_project/runs/AB_prompt_validation_20261022'
PYTHON = '/path/to/ece285_math_rl_project/.venv/bin/python'
RELEASE = '2a0e7781f0c9c31b10edd94ee84ec00eb135c9682ca64dd4eded4318d8853d72'
SSH = ['ssh', '-p', '11032', '-i', '/home/USER/.ssh/fym_125_76_110_84_ed25519',
       '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=12', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
       'USER@GPU_HOST']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def remote(code, timeout=120):
    result = subprocess.run(SSH + [PYTHON + ' -B -c ' + shlex.quote(code)],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-3000:] or f'SSH exit {result.returncode}')
    return json.loads(result.stdout)


def render(raw):
    analysis = json.loads((raw / 'analysis.json').read_text())
    summary = json.loads((raw / 'evaluation_summary.json').read_text())
    protocol = json.loads((raw / 'protocol.json').read_text())
    assert analysis['passed'] and analysis['independently_reverified'] == 3992
    assert analysis['protected_old_files_unchanged'] == 210
    assert not analysis['official_reward_changed'] and not analysis['training']
    assert summary['release_sha256'] == RELEASE and summary['count'] == 3992
    assert summary['predictions_sha256'] == sha(raw / 'predictions.jsonl')
    labels = {'A_original': 'A · 原提示', 'A_final_only': 'A · 末尾答案框',
              'B_original': 'B · 原提示', 'B_final_only': 'B · 末尾答案框'}
    contrast_labels = {'B_minus_A_original': '原提示：B−A',
                       'B_minus_A_final_only': '末尾框提示：B−A',
                       'prompt_effect_A': 'A：新提示−原提示',
                       'prompt_effect_B': 'B：新提示−原提示',
                       'interaction': 'B的提示效应−A的提示效应'}
    sections = []
    notes = []
    for dataset, part in analysis['datasets'].items():
        title = 'MATH 预留验证题' if dataset == 'math_validation' else 'GSM8K 验证题（来自train split）'
        rows = []
        for cell in protocol['cells']:
            m = part['cells'][cell]
            rows.append(f'<tr><td>{labels[cell]}</td><td>{m["correct"]}/{m["n"]} · {100*m["accuracy"]:.2f}%</td>'
                        f'<td>{100*m["single_terminal_box"]/m["n"]:.1f}%</td>'
                        f'<td>{100*m["early_last_box"]/m["n"]:.1f}%</td>'
                        f'<td>{m["truncated"]}</td><td>{m["mean_generated_tokens"]:.0f}</td></tr>')
        comparisons = []
        for key, label in contrast_labels.items():
            v = part['paired_contrasts'][key]
            lo, hi = v['ci95_pp']
            comparisons.append(f'<tr><td>{label}</td><td>{v["point_pp"]:+.2f}</td><td>[{lo:+.2f}, {hi:+.2f}]</td></tr>')
        v = part['paired_contrasts']['B_minus_A_final_only']
        lo, hi = v['ci95_pp']
        if hi < 0:
            interpretation = '在新提示下，B的准确率仍低于A，配对区间未跨0。'
        elif lo > 0:
            interpretation = '在新提示下，B的准确率高于A，配对区间未跨0；这是单seed、探索性比较。'
        else:
            interpretation = '在新提示下，B−A的配对区间跨0，当前数据不能证明B优于A，也不能证明两者等效。'
        notes.append(title + '：' + interpretation)
        sections.append(f'<h2>{title} · n={part["n"]}</h2><div class="scroll"><table><tr><th>组合</th><th>原验答准确率</th>'
                        '<th>单个末尾框</th><th>最后框过早</th><th>截断条数</th><th>平均生成token</th></tr>'
                        + ''.join(rows) + '</table></div><table><tr><th>配对比较</th><th>差值（百分点）</th>'
                        '<th>95%区间</th></tr>' + ''.join(comparisons) + '</table><p>' + interpretation + '</p>')
    done = datetime.now(timezone.utc).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT')
    content = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A/B提示词对照结果</title><style>body{background:#f3f6fa;color:#223247;font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;margin:0}main{max-width:1100px;margin:32px auto;padding:30px;background:white;border:1px solid #dbe4ef;border-radius:14px}h1{font-size:28px}h2{font-size:21px;margin-top:32px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:10px;border-bottom:1px solid #dbe4ef;text-align:left}th{background:#edf4fa}.note{background:#fff6df;padding:15px;border-left:4px solid #bd8518}.ok{background:#e8f5ee;padding:15px}.scroll{overflow-x:auto}a{color:#26698e}code{font-size:13px;overflow-wrap:anywhere}@media(max-width:700px){main{margin:0;padding:16px}table{font-size:13px}}</style></head><body><main>
<h1>现有 A/B checkpoint：提示词对照结果</h1>'''
    content += f'<p class="ok">四格推理完成，3,992条回答已按原规则独立重验；210个受保护旧文件哈希一致。整理时间：{done}。</p>'
    content += '<p>A是只用结果奖励训练的模型，B是加固定7B过程奖励训练的模型。这里仅使用已有seed17、约6M token checkpoint做推理，比较原提示与“先推导、末尾一个答案框”提示。</p>'
    content += '<p class="note"><strong>解释边界：</strong>“单个末尾框”只衡量输出结构，不能证明正文推导与答案一致或推导正确。固定的12题×4回答盲化阅读材料已保存，定性核对尚待完成。提示词也可能改变推理正文，不能把准确率变化全归因于答案框位置。</p>'
    content += ''.join(sections)
    content += '<h2>控制与限制</h2><ul><li>两种提示均用于A和B；同一题、greedy、2048生成token上限、同一验答实现。保留全部结果，没有按结果筛题或重选提示。</li><li>498道MATH预留题、500道GSM8K train split题，与当前RL训练题和原测试集没有规范化文本重复；不能保证基础模型预训练未见过。MATH验证题与公开PRM800K题库重合，不能当作PRM无污染泛化评测。</li><li>区间由10,000次同题配对bootstrap计算，只覆盖题目采样的不确定性；没有覆盖训练seed变动。多项比较为探索性、未做多重比较校正。</li><li>共享GPU1，推理显存预算25%。仅生成前的空闲显存启动门槛由40,000改为30,000MiB；旧代码和release已归档，生成参数、数据、模型和验答规则不变。共享吞吐不作为模型效率结论。</li><li>原MATH500和GSM8K测试成绩保持原样，本页验证集分数不替换原测试结果。</li></ul>'
    content += f'<p>本轮生成 {summary["generated_completion_tokens"]:,} 个completion token。推理总用时 {summary["elapsed_seconds"]/3600:.2f} 小时。</p>'
    content += '<p><a href="final_evidence/analysis.json">完整配对分析</a> · <a href="final_evidence/evaluation_summary.json">推理完成记录</a> · <a href="final_evidence/blind_review.txt">盲化核对材料（先不看key）</a> · <a href="final_evidence/protocol.json">固定协议</a> · <a href="CONTINUATION.txt">续接记录</a></p></main></body></html>'
    if not (OUT / 'review_before_results.html').exists():
        (OUT / 'review_before_results.html').write_bytes((OUT / 'review.html').read_bytes())
    (OUT / 'review.html').write_text(content)
    atomic(OUT / 'STATUS.json', {'status': 'quantitative_analysis_complete_qualitative_review_pending',
           'updated_utc': datetime.now(timezone.utc).isoformat(), 'training': False,
           'official_results_ready': True, 'results_scope': 'new_validation_only', 'release_sha256': RELEASE,
           'completed_predictions': 3992, 'original_results_preserved': True,
           'qualitative_review_complete': False, 'interpretation': notes})


def main():
    lock = (OUT / 'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'collector.json').exists(), 'Inspect existing collector; do not duplicate'
    state = {'status': 'waiting_for_existing_controller', 'pid': os.getpid(),
             'script_sha256': sha(__file__), 'started_unix': time.time(), 'release_sha256': RELEASE,
             'new_training': False, 'can_launch_gpu_work': False, 'maximum_wait_hours': 48}
    def record(**kw):
        state.update(kw, heartbeat_unix=time.time())
        atomic(OUT / 'collector.json', state)
    record()
    try:
        query = f"from pathlib import Path;import json;p=Path({REMOTE!r});print(json.dumps({{'controller':json.loads((p/'controller.json').read_text()),'state':json.loads((p/'state.json').read_text())}}))"
        while True:
            try:
                current = remote(query)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                record(last_connection_error=str(exc))
                if time.time() - state['started_unix'] >= 48 * 3600:
                    raise RuntimeError('Collection deadline reached; remote job not stopped') from exc
                time.sleep(60)
                continue
            c = current['controller']
            assert c['pid'] == 224797 and c['release_sha256'] == RELEASE, 'Unexpected controller; no action'
            assert current['state']['release_sha256'] == RELEASE
            record(remote_status=c['status'], completed_predictions=current['state'].get('completed_predictions', 0),
                   last_connection_error=None)
            if c['status'] == 'failed':
                raise RuntimeError('Remote controller failed; no restart: ' + str(c.get('error')))
            if c['status'] == 'completed':
                break
            if time.time() - state['started_unix'] >= 48 * 3600:
                raise RuntimeError('Collection deadline reached; remote job not stopped')
            time.sleep(60)
        record(status='collecting_completed_results')
        bundle_code = f'''
from pathlib import Path
import hashlib,json,tarfile
p=Path({REMOTE!r})
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
c=json.loads((p/'controller.json').read_text());a=json.loads((p/'analysis.json').read_text())
assert c['status']=='completed' and c['release_sha256']=={RELEASE!r}
assert a['passed'] and a['independently_reverified']==3992
assert sha(p/'predictions.jsonl')==a['predictions_sha256']
bundle=p/'completed_results.tar.gz';manifest=p/'handoff_manifest.json'
assert not bundle.exists() and not manifest.exists(),'Inspect existing bundle; no overwrite'
files=sorted(f for f in p.rglob('*') if f.is_file())
assert not any(f.is_symlink() for f in files)
data={{str(f.relative_to(p)):sha(f) for f in files}}
with manifest.open('x') as f:json.dump(data,f,indent=2)
with tarfile.open(bundle,'x:gz') as tar:
    for f in files+[manifest]:tar.add(f,arcname=str(f.relative_to(p)),recursive=False)
print(json.dumps({{'bundle_sha256':sha(bundle),'file_count':len(data),'bundle_bytes':bundle.stat().st_size}}))
'''
        info = remote(bundle_code, timeout=300)
        partial = OUT / 'completed_results.tar.gz.part'
        assert not partial.exists() and not (OUT / 'completed_results.tar.gz').exists()
        with partial.open('xb') as f:
            result = subprocess.run(SSH + ['cat ' + shlex.quote(REMOTE + '/completed_results.tar.gz')],
                                    stdout=f, stderr=subprocess.PIPE, timeout=900)
        assert result.returncode == 0, result.stderr.decode(errors='replace')[-2000:]
        assert sha(partial) == info['bundle_sha256']
        final = OUT / 'completed_results.tar.gz';partial.rename(final)
        raw = OUT / 'final_evidence';raw.mkdir(exist_ok=False)
        with tarfile.open(final, 'r:gz') as tar:
            for member in tar.getmembers():
                parts = PurePosixPath(member.name)
                assert not parts.is_absolute() and '..' not in parts.parts and member.isfile()
            tar.extractall(raw)
        manifest = json.loads((raw / 'handoff_manifest.json').read_text())
        assert len(manifest) == info['file_count']
        for name, value in manifest.items():
            assert sha(raw / name) == value, name
        assert sha(raw / 'release.json') == RELEASE
        atomic(OUT / 'final_transfer.json', dict(info, all_member_hashes_passed=True,
               finished_utc=datetime.now(timezone.utc).isoformat()))
        render(raw)
        with (OUT / 'CONTINUATION.txt').open('a') as f:
            f.write('\nQUANTITATIVE RESULTS COLLECTED: final_evidence contains verified completed run and analysis; review.html updated automatically. Fixed48-response qualitative blind review is still pending. Read blind_review.txt before blind_review_key.json; independent question solutions already saved locally. Never rerun inference.\n')
        record(status='completed_quantitative_report', finished_unix=time.time(),
               qualitative_review_complete=False, all_downloaded_hashes_passed=True)
    except BaseException as exc:
        record(status='failed', error=str(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        raise


if __name__ == '__main__':
    main()
