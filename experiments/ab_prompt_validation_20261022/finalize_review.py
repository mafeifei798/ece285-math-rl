"""Finish the local report from immutable inference results and sealed review.

No GPU work, new grading rule, model update, or editing of downloaded evidence.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import html
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/ab_prompt_validation_20261022'
RAW = OUT / 'final_evidence'
CELLS = ['A_original', 'A_final_only', 'B_original', 'B_final_only']
NAMES = dict(zip(CELLS, ['A · 原提示', 'A · 末尾框提示', 'B · 原提示', 'B · 末尾框提示']))
LABELS = {'agree':'末框与最后正文结论相同', 'disagree':'明确不一致', 'unresolved':'未形成稳定结论', 'no_valid_box':'无有效答案框'}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_new(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    old_report = ROOT / 'execution/ab_corrected_6m_seed17_20261022/review.html'
    old_hash = sha(old_report)
    before = {name: sha(RAW/name) for name in json.loads((RAW/'handoff_manifest.json').read_text())}
    assert before == json.loads((RAW/'handoff_manifest.json').read_text())
    analysis = json.loads((RAW/'analysis.json').read_text())
    evaluation = json.loads((RAW/'evaluation_summary.json').read_text())
    review = json.loads((OUT/'qualitative_review_unblinded.json').read_text())
    failure = json.loads((OUT/'supplementary_failure_analysis.json').read_text())
    blind = {r['review_id']:r for r in json.loads((RAW/'blind_review.json').read_text())}
    sealed = OUT/'qualitative_annotations_blinded_v1.json'
    assert sha(sealed) == review['blinded_annotations_sha256'] == '579720d8d7069f1247ec639c60fb1aef30be7b7a61beb68504ca8f23a7697f80'
    rows = [json.loads(x) for x in (RAW/'predictions.jsonl').read_text().splitlines()]
    assert len(rows) == len({(r['cell'],r['problem_id']) for r in rows}) == 3992
    assert sha(RAW/'predictions.jsonl') == evaluation['predictions_sha256'] == analysis['predictions_sha256']
    for ds, data in analysis['datasets'].items():
        for cell in CELLS:
            subset = [r for r in rows if r['dataset']==ds and r['cell']==cell]
            assert len(subset) == data['n']
            assert int(sum(r['reward'] for r in subset)) == data['cells'][cell]['correct']
            assert sum(failure['datasets'][ds]['cells'][cell].values()) == data['n']
    now = datetime.now(timezone.utc)
    stamp = now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M PT')
    conclusion = [
        '本轮四组推理已完成。新提示下B仍低于A：MATH差4.22个百分点，GSM8K差8.60个百分点。',
        '新提示提高了单个末尾框的结构合规率，但没有建立数学正文一致性已经改善的证据；不能将本轮解释为排除格式问题后的纯数学能力比较。',
        '固定48条助理定性审读中，最后正文结论与末框相同的数量：A为8/12到7/12，B为9/12到8/12。明确冲突虽减少，未形成稳定结论的回答增多。',
        'GSM8K上新提示使A下降8.4个百分点、B下降14.6个百分点；B对这一次提示变化更敏感。该结论限于本轮checkpoint、验证题和固定提示对。',
        '保留原提示结果作为主报告；本次提示干预作为未能修复B表现的负结果，不替换原测试成绩，不自动搜索新提示或启动训练。',
    ]
    result = {
        'created_utc':now.isoformat(), 'status':'complete_with_qualitative_review',
        'generation_finished_utc':evaluation['finished_utc'], 'predictions':3992,
        'official_grading_changed':False, 'training':False,
        'official_results_scope':'498 MATH held-out validation questions and 500 GSM8K train-split held-out questions; not the original test scores',
        'datasets':analysis['datasets'], 'qualitative_summary':review['summary'],
        'conclusions_zh':conclusion,
        'limitations':analysis['limitations'] + review['limitations'],
        'hashes':{name:sha(OUT/name) for name in ['qualitative_annotations_blinded_v1.json','qualitative_review_unblinded.json','supplementary_failure_analysis.json']},
        'predictions_sha256':sha(RAW/'predictions.jsonl'),
    }
    write_new(OUT/'final_analysis_summary.json', result)

    # Preserve the automatic quantitative-only report, then append the completed
    # qualitative audit and the interpretation rather than altering any result.
    page = OUT/'review.html'
    original = page.read_text()
    with (OUT/'review_quantitative_only.html').open('x') as f:
        f.write(original)
    qualrows = ''
    for cell in CELLS:
        counts = review['summary'][cell]['consistency']
        qualrows += '<tr><td>'+NAMES[cell]+'</td>'+''.join('<td>'+str(counts.get(k,0))+'/12</td>' for k in ['agree','disagree','unresolved','no_valid_box'])+'</tr>'
    overview = '''<h2>结论：这次提示干预没有救回 B</h2>
<p><strong>新提示下，B 仍低于 A；但不能声称“答案框与正文一致性已修好，所以剩下全是数学能力差距”。</strong>本次提示同时改变了输出结构和解题内容，一致性修复没有成功建立。</p>
<p>GSM8K：原提示下 A 为76.0%、B为73.6%；新提示下 A 为67.6%、B为59.0%。B−A从−2.4扩大到−8.6个百分点。提示效应的模型间差异为−6.2个百分点，探索性配对95%区间[−10.8, −1.6]。这些是固定seed17 checkpoint的题目层面结果。</p>
<p>这次不采用新提示替换原评测提示。保留原A/B成绩，记录本轮负结果；没有更改验答、重新训练或补跑其他提示。</p>
<h2>答案框是否真的更一致：固定48条定性核对</h2>
<p>在推理前选定12题（4道MATH、8道GSM8K），每题四种回答。助理隐藏组名逐条阅读，先保存标签及哈希，再读取组别key。已见过整组准确率；提示风格可能透露分组，因此不宣称完全盲法。这不是新增人工标注或独立judge评分。</p>
<div class="scroll"><table><tr><th>组合</th><th>末框与最后正文结论相同</th><th>明确不一致</th><th>无稳定结论</th><th>无有效框</th></tr>'''+qualrows+'''</table></div>
<p>明确冲突的条数降低，伴随重复、截断和互相矛盾的正文增多；“能明确对齐”的条数没有增加。只有12题，不能把这些比例外推到全验证集。“结论相同”也不等于推导正确，例如先算16800又无依据改口18000，框18000仍会被标为最后结论相同。</p>
<p><a href="qualitative_cases.html">查看全部48条原文、证据与标签</a> · <a href="qualitative_annotations_blinded_v1.json">解盲前封存标签</a> · <a href="qualitative_review_unblinded.json">解盲后的分组结果</a></p>
<h2>固定样本中的具体变化</h2>
<ul><li><strong>Q08 纱线题：</strong>正确总长632cm。B原提示框506、正文632；B新提示框566、正文仍明确说632。冲突持续存在。A新提示在这题修正为632。</li>
<li><strong>Q10 Zoom时长题：</strong>A/B原提示都算出并最终框出82分钟；新提示下两者遗漏14分钟故障时间，反复算68，开头框100，最后均达到2048 token上限。这一次干预改变了推导内容。</li>
<li><strong>Q09 收入题：</strong>B新提示多次算得16800，随后无依据改口18000，最后正文和框都错成18000。表面一致不能证明数学正确。</li>
<li><strong>Q12 漫画题：</strong>新提示也确实修好了一些题：两模型从逐年取整的728改为正确的730。完整统计同时保留改善与恶化，没有挑例子替代总体结果。</li></ul>
<h2>准确率下降由哪些输出状态构成</h2>
<p>GSM8K中，B有118题从对变错、45题从错变对，净少答对73题。118条恶化里，69条仍可提取但数值不正确，35条缺失/无效答案框，14条截断。<strong>可提取但数值错误，不等于已经证明正文推导错误</strong>，其中仍可能包含框与正文冲突。</p>
<div class="scroll"><table><tr><th>GSM8K组别</th><th>答对</th><th>可提取但答错</th><th>无有效框</th><th>截断</th><th>无法解析内容</th></tr>'''
    for cell in CELLS:
        c = failure['datasets']['gsm8k_validation']['cells'][cell]
        overview += '<tr><td>'+NAMES[cell]+'</td>'+''.join('<td>'+str(c.get(k,0))+'</td>' for k in ['correct','parseable_incorrect_answer','missing_or_invalid_box','truncated','unparseable'])+'</tr>'
    overview += '</table></div><p><a href="supplementary_failure_analysis.json">全部状态分解及同题转移</a> · <a href="final_analysis_summary.json">最终分析JSON</a></p>'
    needle = '<h2>MATH 预留验证题'
    assert original.count(needle)==1
    page_text = original.replace(needle, overview+needle)
    page_text = page_text.replace('固定的12题×4回答盲化阅读材料已保存，定性核对尚待完成。', '固定的12题×4回答定性审读已完成，见下表；样本很小，不能据此估计总体一致率。')
    page_text = page_text.replace('整理时间：2026-10-22 20:48 PT。', '定量及定性整理完成：'+stamp+'。')
    page.write_text(page_text)

    header = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>48条回答一致性审读</title><style>body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#223247;background:#f3f6fa}main{max-width:1080px;margin:24px auto;padding:28px;background:#fff}details{border:1px solid #ccd7e4;border-radius:8px;padding:12px;margin:14px 0}summary{font-weight:600;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace;background:#f6f8fa;padding:14px}.note{background:#fff6df;padding:14px}a{color:#26698e}</style><main><h1>固定48条回答：答案框与正文审读</h1><p><a href="review.html">返回四组总报告</a></p><p class="note">助理定性审读，不是人类标注或额外模型judge。只读生成代码和声称的输出；未执行回答中的代码。先封存标签，再解盲；整组准确率此前已知。12题的描述性检查不能代表全部验证题。</p>'
    details = []
    for r in review['joined_annotations']:
        b = blind[r['review_id']]
        details.append('<details><summary>'+html.escape(r['review_id']+' · '+NAMES[r['cell']]+' · '+LABELS[r['box_body_consistency']])+'</summary>'
            +'<p>原验答：'+str(int(r['reward']))+'；结束状态：'+html.escape(r['finish_reason'])+'</p>'
            +'<p><strong>判读：</strong>'+html.escape(r['reviewer_note_zh'])+'</p>'
            +'<p>最后正文结果：'+html.escape(str(r['last_body_result']))+'；最后有效框：'+html.escape(str(r['last_valid_box']))+'</p>'
            +'<p>核对片段：</p><pre>'+html.escape(r['evidence_excerpt'])+'</pre>'
            +'<h3>题目</h3><pre>'+html.escape(b['problem'])+'</pre><h3>完整回答</h3><pre>'+html.escape(b['completion'])+'</pre></details>')
    with (OUT/'qualitative_cases.html').open('x') as f:
        f.write(header+''.join(details)+'</main></html>')

    status = json.loads((OUT/'STATUS.json').read_text())
    write_new(OUT/'STATUS_quantitative_only.json', status)
    status.update(status='analysis_complete_with_qualitative_review', updated_utc=now.isoformat(), qualitative_review_complete=True, interpretation=conclusion)
    (OUT/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    assert old_hash == sha(old_report)
    assert all(sha(RAW/name)==h for name,h in before.items())
    audit = {'created_utc':now.isoformat(),'passed':True,'downloaded_files_unchanged':len(before),
        'official_rows':len(rows),'official_counts_recomputed':True,'all_998_prompt_pairs_match_frozen_templates':failure['all_998_prompt_pairs_match_frozen_templates'],
        'sealed_review_sha256':sha(sealed),'old_AB_report_unchanged':True,'old_AB_report_sha256':old_hash,
        'new_generation':False,'new_training':False,'new_grading_rule':False,
        'outputs':{name:sha(OUT/name) for name in ['review.html','qualitative_cases.html','final_analysis_summary.json','qualitative_review_unblinded.json','supplementary_failure_analysis.json']}}
    write_new(OUT/'final_review_acceptance.json', audit)
    with (OUT/'CONTINUATION.txt').open('a') as f:
        f.write('\nFINAL ANALYSIS '+now.isoformat()+': Existing controller completed evaluate+analyze at2026-10-23 03:47UTC (Oct22 20:47PT);3992 outputs and2176338 completiontokens. All35 downloaded manifestmembers rehashed; old210files verifiedbyfrozenanalysis. No furtherGPUworkortraining. Fixed48response qualitative labels sealed SHA'+sha(sealed)+' beforekeyread, thenunblinded; aggregateaccuraciesalreadyseen. Completedreview.html, qualitative_cases.html, final_analysis_summary.json, supplementary_failure_analysis.json and final_review_acceptance.json. Newpromptdoesnotestablishsemanticconsistencyimprovement: Aagree8to7/12,B9to8/12; moreunresolvedresponses. B-A withnewprompt:MATH-4.2169ppCI[-7.8313,-0.8032];GSM-8.6ppCI[-12.2,-5.0]. GSMpromptloss A-8.4pp,B-14.6pp. Do not reinterpret as formattingcontrolledpuremathdifference; keepoldpromptandoldtestscores. CollectorhascompletedandnoGPUprocessremainsfor thisrun.\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
