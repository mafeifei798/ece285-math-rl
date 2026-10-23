"""Record assistant qualitative judgments before opening the cell assignment key.

This never grades official accuracy, executes generated code, or launches inference.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'execution/ab_prompt_validation_20261022'
SOURCE = OUT / 'final_evidence/blind_review.json'

# For each hidden response: consistency, last body result, final box, final
# mathematical computation, exact evidence substring, reviewer explanation.
CASES = {
 'Q01': [
  ('agree','4','4','correct','x = 4','分数化简和倒数运算正确；开头空框不影响末框4。'),
  ('agree','4','4','correct','x = 4','正文算得4，末框4。'),
  ('agree','4','4','correct','x = 4','正文算得4，末框4。'),
  ('agree','4','4','correct','indeed \\(4\\)','正文及代入检查均支持4。'),
 ],
 'Q02': [
  ('unresolved','240 / 60, then retracted','69 degrees','incorrect_incomplete','x = 240','图形关系解释错误，反复推翻并截断，未接受最终正文结论。'),
  ('unresolved','240, repeatedly reconsidered','61 degrees','incorrect_incomplete','x = 240','多边形假设反复变换并截断，不能把临时240当成稳定结论。'),
  ('unresolved','240, repeatedly reconsidered','60','incorrect_incomplete','x = 240','推导240后进入重复段落直至截断。'),
  ('disagree','240 degrees','127 degrees','incorrect','= 240^\\circ','正文两次明确给240，末框127没有相应推导；独立正确答案为120。'),
 ],
 'Q03': [
  ('agree','1','1','incorrect','the GCD of $169$ and $a+13$','错误多项式展开和整除判断，正文结论与框同为1。'),
  ('agree','13','13','incorrect','not necessarily even','忽略k为奇数导致91k+1必为偶数；正文与框一致但答案应为26。'),
  ('disagree','1 in the final unboxed conclusion','13','incorrect','the only common divisor of \\(a + 13\\) and \\(16\\) is \\(1\\)','余数误写16，正文最后断言公因数1，却框出13。'),
  ('agree','13','13','incorrect','the GCD must be \\(13\\)','余数、奇偶和整除逻辑均有错；最后正文断言13与框一致。'),
 ],
 'Q04': [
  ('agree','1201','1201','incorrect','```output\n1201','展示的所谓代码输出1201与框一致，但独立精确运算为8；未运行生成代码。'),
  ('agree','1201','1201','incorrect','```output\n1201','展示输出与框一致，但算术结果错误。'),
  ('agree','601','601','incorrect','```output\n601','展示输出与框一致，但独立精确结果为8。'),
  ('agree','601','601','incorrect','```output\n601','展示输出与框一致，但独立精确结果为8。'),
 ],
 'Q05': [
  ('agree','274','274','correct','264 + 10 = 274','开头300在后文被纠正，实际末框274与正文一致。'),
  ('agree','274','274','correct','264 + 10 = 274','开头空框，正文与有效末框均为274。'),
  ('agree','274','274','correct','K = 274','开头300被后文与末框274纠正。'),
  ('agree','274','274','correct','264 + 10 = 274','末框274与计算一致；后附代码没有改变最终数值。'),
 ],
 'Q06': [
  ('no_valid_box','10',None,'correct','there are 10 grey cats','算对10但完全没有答案框。'),
  ('agree','10','10','correct','12 - 2 = 10','计算正确，末框10。'),
  ('no_valid_box','10',None,'correct','there are 10 grey cats','仅有开头空框，后文正确结论10没有装入框内。'),
  ('agree','10','10','correct','16 - 2 - 4 = 10','框在开头，但与正文计算10一致。'),
 ],
 'Q07': [
  ('agree','36000','36000','correct','90 \\times 400 = 36000','总手机90部，每部400，正确末框36000。'),
  ('agree','36000','36000','correct','90 \\times 400 = 36000','计算与末框一致。'),
  ('agree','36000','36000','correct','90 \\times 400 = 36000','计算与末框一致。'),
  ('agree','36000','36000','correct','90 \\times 400 = 36000','开头8400后被末框36000纠正。'),
 ],
 'Q08': [
  ('disagree','632','566','correct','total length of the two pieces of yarn should be 632','计算632正确，唯一框566未更新；捏造题面提供566。'),
  ('disagree','632','506','correct','Total length = 632 cm','计算632但又把506放进最后框，并错误归因于题面。'),
  ('disagree','632','566','correct','The correct total length is 632 cm, not 566 cm.','明确纠正为632，却保留唯一框566。'),
  ('agree','632','632','correct','```output\n632','计算和最终框均为632。'),
 ],
 'Q09': [
  ('disagree','16800','12000','correct','the correct answer is $16800','正确计算16800，唯一框12000未更新。'),
  ('agree','16800','16800','correct','```output\n16800.0','收入公式正确，末框16800。'),
  ('agree','18000 asserted after rejecting 16800','18000','incorrect','The correct answer is $18000','数次算对16800后无依据否定，最终正文断言18000。框与最后断言一致，不代表推导一致或正确。'),
  ('disagree','16800','10800','correct','The correct combined income of Ernie and Jack is $16800.','正文明确否定10800并给16800，但框未更新。'),
 ],
 'Q10': [
  ('agree','82','82','correct','40 + 42 = 82','开头100被正确计算和末框82纠正。'),
  ('agree','82','82','correct','40 + 42 = 82','开头100被末框82纠正。'),
  ('unresolved','68, repeatedly reconsidered','100','incorrect_incomplete','40 + 28 = 68','遗漏14分钟故障时间，重复自我纠正并截断，没有稳定最终结论。'),
  ('unresolved','68, repeatedly reconsidered','100','incorrect_incomplete','40 + 28 = 68','遗漏故障时长，循环直至截断。'),
 ],
 'Q11': [
  ('agree','114','114','correct','15 + 30 + 25 + 33 + 11 = 114','正文计算和末框114一致，开头120已纠正。'),
  ('unresolved','114 in arithmetic / 104 in claimed output','100','internally_conflicting','Reach max function call limit.','正文正确计算114，伪输出反复104，结束于生成的调用上限文字；没有真实工具调用。'),
  ('agree','114','114','correct','15 + 30 + 25 + 33 + 11 = 114','末框114与正确计算一致。'),
  ('unresolved','sum implies 114 / claimed output 104','105','internally_conflicting','Reach max function call limit.','动物数量正确，伪输出104与框105和实际总数114均不一致，未形成稳定终结。'),
 ],
 'Q12': [
  ('agree','730','730','correct','1460 / 2 = 730','最终计算4*365/2正确；开头周末提法无依据但未用于最终计算。'),
  ('agree','728','728','incorrect','182 \\text{ comics/year} \\times 4','逐年向下取整错误，正文和框同错为728。'),
  ('agree','728','728','incorrect','182 \\text{ comics/year} \\times 4','逐年舍弃半天累计偏差；不是答案框冲突。'),
  ('agree','730','730','correct','1460 / 2 = 730','最终计算与框一致，周末措辞未用于实际算式。'),
 ],
}

def main():
    target = OUT / 'qualitative_annotations_blinded_v1.json'
    assert not target.exists(), 'Never silently overwrite sealed judgments'
    blind = json.loads(SOURCE.read_text())
    annotations = []
    for row in blind:
        qid, rid = row['review_id'].split('-')
        cons, body, box, quality, evidence, note = CASES[qid][int(rid[1:])-1]
        assert evidence in row['completion'], (row['review_id'], evidence)
        annotations.append({
            'review_id': row['review_id'], 'box_body_consistency': cons,
            'last_body_result': body, 'last_valid_box': box,
            'final_mathematical_computation': quality,
            'finish_reason': row['finish_reason'], 'evidence_excerpt': evidence,
            'reviewer_note_zh': note,
            'response_sha256': hashlib.sha256(row['completion'].encode()).hexdigest(),
        })
    assert len(annotations) == len({r['review_id'] for r in annotations}) == 48
    payload = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Assistant qualitative reading; not human labels and no external judge calls.',
        'assignment_key_read_before_sealing': False,
        'aggregate_accuracy_seen_before_review': True,
        'sample': 'Preselected 12 questions x 4 cells (4 MATH, 8 GSM8K). Descriptive only.',
        'rubric': {
            'agree': 'Final valid box equals the final accepted unboxed body result, including explicitly claimed output. This does not certify the derivation.',
            'disagree': 'Final valid box differs from a clear accepted body result without a supported correction.',
            'unresolved': 'Truncated reconsideration or contradictory body results do not yield a stable accepted conclusion.',
            'no_valid_box': 'No nonempty valid box exists even if the body reaches a correct answer.',
        },
        'caveats': ['Full chain validity is not exhaustively scored.',
                    'Early wrong boxes corrected by the final box are not final box-body disagreements.',
                    'Generated code and claimed tool outputs were read as text, never executed.',
                    'Prompt style can suggest assignment; hiding group labels does not ensure perfect reviewer blinding.'],
        'blind_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256((OUT/'blind_independent_reference.json').read_bytes()).hexdigest(),
        'annotations': annotations,
    }
    with target.open('x') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({'path':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'count':48}))

if __name__ == '__main__':
    main()
