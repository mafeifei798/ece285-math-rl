"""CPU-only, post-hoc output-order audit. Does not regrade or load any model."""
from pathlib import Path
import hashlib,json,re,statistics
from collections import Counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPEC={'post_hoc_descriptive_only':True,'first_box_relative_position':'decoded character start / total decoded characters',
      'early_relative_threshold':.1,'leading_prefix_nonwhitespace_characters':20,
      'minimum_post_box_nonwhitespace_characters':100,
      'placeholder_contents':['','?','.','...','…',r'\dots',r'\ldots',r'\cdots','answer','unknown','tbd','n/a'],
      'primary_files':'seed17:initial+6M A+6M C-bounded, same benchmark question IDs; seed42 A/C as descriptive replication',
      'no_causal_faithfulness_claim':True,'no_new_training_or_PRM_calls':True}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.open() if s.strip()]
def compact(s):return re.sub(r'\s+','',s)
def escaped(text,i):
    n=0;i-=1
    while i>=0 and text[i]=='\\':n+=1;i-=1
    return n%2==1

def boxes(text):
    result=[]
    for m in re.finditer(r'\\boxed\s*\{',text):
        start=m.end()-1;depth=0;end=None
        for i in range(start,len(text)):
            if escaped(text,i):continue
            if text[i]=='{':depth+=1
            elif text[i]=='}':
                depth-=1
                if depth==0:end=i+1;break
        content=text[start+1:end-1] if end is not None else None
        placeholder=content is not None and compact(content).lower() in SPEC['placeholder_contents']
        result.append({'start':m.start(),'end':end,'content':content,'placeholder':placeholder})
    return result

def describe(record):
    text=record['completion'];bb=boxes(text);complete=[b for b in bb if b['end'] is not None]
    first=bb[0] if bb else None
    values={'reward':int(record['reward']),'has_marker':bool(bb),'complete_boxes':len(complete),
        'first_marker_position_fraction':first['start']/max(1,len(text)) if first else None,
        'first_marker_in_first10pct':bool(first and first['start']/max(1,len(text))<=.1),
        'first_marker_in_first20nonwhitechars':bool(first and len(compact(text[:first['start']]))<=20),
        'first_box_incomplete':bool(first and first['end'] is None),
        'first_box_placeholder':bool(first and first['end'] is not None and first['placeholder']),
        'first_box_nonplaceholder':bool(first and first['end'] is not None and not first['placeholder']),
        'finish_reason':record.get('finish_reason')}
    body=bool(first and first['end'] is not None and len(compact(text[first['end']:]))>=100)
    leading=values['first_marker_in_first20nonwhitechars'] and body
    values.update(first_box_followed_by_at_least100_nonwhitechars=body,
        leading_placeholder_box_with_body=leading and values['first_box_placeholder'],
        leading_nonplaceholder_box_with_body=leading and values['first_box_nonplaceholder'],
        early10pct_nonplaceholder_box_with_body=values['first_marker_in_first10pct'] and body and values['first_box_nonplaceholder'])
    nonplaceholder=[b for b in complete if not b['placeholder']]
    values['first_last_nonplaceholder_box_text_equal']=bool(len(nonplaceholder)>=2 and compact(nonplaceholder[0]['content'])==compact(nonplaceholder[-1]['content']))
    values['two_or_more_nonplaceholder_boxes']=len(nonplaceholder)>=2
    return values

def summarize(data):
    desc=[describe(r) for r in data];n=len(data)
    assert all(int(r['generated_tokens'])==len(r['completion_token_ids']) for r in data)
    binary=[k for k,v in desc[0].items() if isinstance(v,bool)]
    counts={k:sum(d[k] for d in desc) for k in binary}
    pos=[d['first_marker_position_fraction'] for d in desc if d['has_marker']]
    grouped={}
    for key in ['leading_nonplaceholder_box_with_body','leading_placeholder_box_with_body']:
        for value in [True,False]:
            group=[d for d in desc if d[key]==value]
            grouped[key+'='+str(value)]={'n':len(group),'correct':sum(d['reward'] for d in group),
                'accuracy':sum(d['reward'] for d in group)/len(group) if group else None}
    return {'n':n,'correct':sum(d['reward'] for d in desc),'counts':counts,
        'rates':{k:v/n for k,v in counts.items()},'first_box_median_character_fraction':statistics.median(pos) if pos else None,
        'conditional_answer_accuracy':grouped,
        'PRM_scores_present_in_prediction_records':sum('step_scores' in r or 'min_score' in r for r in data)},desc

def main():
    assert boxes(r'x \boxed{\frac{1}{2}} tail')[0]['content']==r'\frac{1}{2}'
    assert boxes(r'\boxed{\{1,2\}}')[0]['content']==r'\{1,2\}'
    assert boxes(r'\boxed{?}')[0]['placeholder']
    assert boxes(r'\boxed{12')[0]['end'] is None
    HERE.mkdir(exist_ok=True)
    spec_path=HERE/'analysis_spec.json'
    if spec_path.exists():assert read(spec_path)==SPEC
    else:spec_path.write_text(json.dumps(SPEC,indent=2)+'\n')
    initial=ROOT/'execution/final_four_eval_20261029/evidence/runs/Final_four_eval_seed17_1M_20261029/initial'
    base17=ROOT/'execution/abc_seed17_resumed_v1_20261104/completed_evidence'
    base42=ROOT/'execution/ac_seed42_v1_20261106/completed_evidence'
    expected_initial=read(ROOT/'execution/final_four_eval_20261029/comparison.json')['summary']['initial']
    expected17=read(base17/'evaluation/288/summary.json')['summary']
    expected42=read(base42/'evaluation/288/summary.json')['summary']
    manifests={};summaries={};all_desc={};all_rows={}
    for seed,arm,directory in [('initial','initial',initial),('17','A',base17/'evaluation/288/A'),
         ('17','C_bounded',base17/'evaluation/288/C_bounded'),('42','A',base42/'evaluation/288/A'),
         ('42','C_bounded',base42/'evaluation/288/C_bounded')]:
        for dataset in ('math500','gsm8k'):
            path=directory/(dataset+'.jsonl');data=rows(path)
            assert len(data)==(500 if dataset=='math500' else 1319)
            assert all(r['reward'] in (0,1) for r in data)
            expected=expected_initial[dataset] if seed=='initial' else (expected17 if seed=='17' else expected42)[arm][dataset]
            assert digest(path)==expected['predictions_sha256']
            assert sum(r['reward'] for r in data)==expected['correct']
            name=f'{seed}/{arm}/{dataset}'
            summaries[name],all_desc[name]=summarize(data);all_rows[name]=data
            manifests[str(path.relative_to(ROOT))]={'sha256':digest(path),'records':len(data)}
    paired={}
    for seed in ('17','42'):
        for dataset in ('math500','gsm8k'):
            a=f'{seed}/A/{dataset}';c=f'{seed}/C_bounded/{dataset}';init=f'initial/initial/{dataset}'
            key=lambda r:(r['problem_id'],r['question_sha256'],r['gold'])
            assert list(map(key,all_rows[a]))==list(map(key,all_rows[c]))==list(map(key,all_rows[init]))
            paired[f'{seed}/{dataset}']={}
            for metric in ('first_marker_in_first20nonwhitechars','leading_nonplaceholder_box_with_body','leading_placeholder_box_with_body'):
                l=[d[metric] for d in all_desc[c]];r=[d[metric] for d in all_desc[a]]
                paired[f'{seed}/{dataset}'][metric]={'C_minus_A_pp':100*(sum(l)-sum(r))/len(l),
                    'C_only':sum(x and not y for x,y in zip(l,r)),'A_only':sum(y and not x for x,y in zip(l,r))}
    training={}
    training_data_path=ROOT/'execution/final_four_eval_20261029/evidence/data/prepared/train.jsonl'
    training_ids={r['problem_id'] for r in rows(training_data_path)}
    assert len(training_ids)==6993
    manifests[str(training_data_path.relative_to(ROOT))]={'sha256':digest(training_data_path),'distinct_question_ids':len(training_ids)}
    previous_order=None
    for arm in ('A','C_bounded'):
        path=base17/arm/'generation_checks.jsonl';rr=rows(path)
        ids=[q for b in rr for q in b['prompt_groups']]
        assert len(rr)==288 and len(ids)==1152
        assert set(ids)<=training_ids
        if previous_order is not None:assert ids==previous_order
        previous_order=ids
        training[arm]={'updates':len(rr),'question_occurrences':len(ids),'distinct_question_ids':len(set(ids)),
            'completions':sum(b['rows'] for b in rr),'observed_unique_fraction_of6993_training_questions':len(set(ids))/len(training_ids)}
        manifests[str(path.relative_to(ROOT))]={'sha256':digest(path),'records':len(rr)}
    cfg={}
    for arm in ('A','C_bounded'):
        path=base17/arm/'resume_build_288.json';x=read(path)['settings']
        cfg[arm]={k:x[k] for k in ('learning_rate','lr_scheduler_type','warmup_steps','num_iterations','gradient_accumulation_steps','generation_batch_size','num_generations','loss_type','beta')}
        assert cfg[arm]['learning_rate']==5e-6 and cfg[arm]['num_iterations']==1
        manifests[str(path.relative_to(ROOT))]={'sha256':digest(path)}
    out={'analysis_spec':SPEC,'settings':cfg,'training_coverage_seed17':training,'summaries':summaries,'paired_format_differences':paired,
        'acceptance':{'passed':True,'prediction_records':sum(v['n'] for v in summaries.values()),
            'prediction_files':10,'original_prediction_hashes_and_accuracy_counts_match':True,
            'matched_question_ids_gold_and_question_hashes':True,'new_optimizer_updates':0,'new_model_or_PRM_calls':0},
        'input_manifest':manifests,'limits':[
        'Output order is observable surface format, not a causal test of faithful mathematical reasoning.',
        'Nonplaceholder means lexical filtering only, not verified mathematical validity of the first box.',
        'Character-relative position is not token-relative position.',
        'Initial and trained predictions come from different evaluation runs; paired data/prompt identities are checked, but numerical repeatability is not exact.',
        'Existing benchmark predictions contain no PRM scores; a controlled comparison of PRM scores requires new scoring and is not completed here.',
        'Conditional accuracy groups differ in difficulty/length and are not randomized interventions.',
        'These are post-hoc descriptive analyses on previously examined test sets, not fresh confirmation.'
        ]}
    (HERE/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    text=['答案框顺序与训练配置：CPU只读审计','实际LoRA学习率5e-6（constant），num_iterations=1；未启动训练或PRM评分。',
        '严格“开头框＋正文”＝首框之前至多20个非空白字符，框后至少100个非空白字符。占位框单独统计；它不是已提交的数学答案。',
        '位置＝解码文本字符位置/全文字符数，不是token位置。',
        'seed / model / dataset: 首框在开头20字符内；非占位首框＋正文；占位首框＋正文；首框相对位置中位数']
    for name,v in summaries.items():
        rr=v['rates'];text.append(f"{name}: {rr['first_marker_in_first20nonwhitechars']*100:.1f}%；{rr['leading_nonplaceholder_box_with_body']*100:.1f}%；{rr['leading_placeholder_box_with_body']*100:.1f}%；{v['first_box_median_character_fraction']*100:.3f}%。")
    text+=['','seed17训练曝光：'+json.dumps(training,ensure_ascii=False),
        '', '这些数字不能证明事后编理由，不能用第一框提前率直接推导faithfulness，也不能把正确率差距归因于RL方法。',
        '所有预测文件均无PRM分数。本次没有补算PRM，因此未回答“后置论证是否被PRM高估”。',
        '见results.json的分组正确率、输入hash及限制。原始预测和验答结果不变。']
    (HERE/'REPORT.txt').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))

if __name__=='__main__':main()
