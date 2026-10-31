"""CPU-only frozen selections, native anchor encodings, and data schedules."""
import collections
import math
import random
from common import *

SOURCE_SHA = 'd9e39ca727f8000c158f3baf51c7fe0c175f86146ad9250c2965f48172006864'


def select():
    assert_root(); verify_release(); OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'selection.json').exists():
        for n, checksum in read(OUT / 'selection.json')['files'].items(): assert sha(OUT / n) == checksum
        return
    protect(create=True)
    train = rows(ROOT / 'data/prepared/train.jsonl'); byid = {r['problem_id']: r for r in train}
    base = rows(ROOT / 'runs/B_corrected_seed17_6M_20261022/rollouts.jsonl')
    groups = collections.defaultdict(list)
    for i in range(0, len(base), 8):
        g = base[i:i+8]; assert len(g) == 8 and len({r['problem_id'] for r in g}) == 1
        n = sum(r['outcome_reward'] for r in g)
        groups['all_wrong' if n == 0 else 'all_correct' if n == 8 else 'mixed'].append(i//8)
    picked = []
    for kind, count in CFG['probe_group_counts'].items():
        chosen = sorted(groups[kind], key=lambda i: rank('probe-' + kind, i))[:count]
        assert len(chosen) == count; picked.extend(chosen)
    probe = []
    for gi in sorted(picked):
        for i in range(gi*8, (gi+1)*8):
            r = base[i]; probe.append({**r, 'question': byid[r['problem_id']]['problem'],
                'question_sha256': byid[r['problem_id']]['question_sha256'], 'source_row': i,
                'group_id': gi, 'example_id': f'B-corrected-{i}'})
    pb = read(ROOT / 'data/raw/processbench_math.json')
    forbidden = {question_key(r['question']) for r in probe}
    for name in ['validation','math500','gsm8k']:
        forbidden.update(r['question_sha256'] for r in rows(ROOT / f'data/prepared/{name}.jsonl'))
    forbidden.update(question_key(r['problem']) for r in pb)
    # Previously examined error groups are disclosed illustrations only.
    old = ROOT / 'experiments/cd_prm_pilot_20261021_gpu/artifacts/probe_B6M.jsonl'
    illustrations = []
    if old.exists():
        selected = [r for r in rows(old) if r.get('illustrative_group')]
        for r in selected:
            forbidden.add(question_key(r['question']))
            illustrations.append({**r, 'example_id': f'old-case-{r["row_index"]}',
                'group_id': f'old-{r["group_index"]}', 'diagnostic_only': True})
    chosen = sorted([r for r in train if r['question_sha256'] not in forbidden], key=lambda r: rank('buffer', r['problem_id']))[:512]
    assert len(chosen) == 512
    buckets = collections.defaultdict(list)
    for r in pb:
        k = 'answer_correct_valid' if r['label'] == -1 else 'answer_correct_invalid' if r['final_answer_correct'] else 'answer_wrong_invalid'
        buckets[k].append(r)
    selected_pb = []
    for key in ['answer_correct_valid','answer_correct_invalid','answer_wrong_invalid']:
        a = sorted(buckets[key], key=lambda r: rank('ProcessBench',r['id']))[:CFG['processbench_per_stratum']]
        assert len(a) == CFG['processbench_per_stratum']
        selected_pb.extend({**r,'stratum':key,'example_id':f'PB-{r["id"]}'} for r in a)
    for name, data in [('buffer_questions.jsonl',chosen),('probe.jsonl',probe),('illustrations.jsonl',illustrations),('processbench.jsonl',selected_pb)]: write_rows(OUT/name,data)
    atomic(OUT/'forbidden.json',sorted(forbidden))
    atomic(OUT/'selection.json',dict(config_sha256=sha(HERE/'config.json'), buffer_questions=512,
        primary_probe_groups=len(picked), primary_probe_rows=len(probe), illustrative_rows=len(illustrations),
        selection='hash-selected within outcome strata, never selected by PRM scores',
        files={n:sha(OUT/n) for n in ['buffer_questions.jsonl','probe.jsonl','illustrations.jsonl','processbench.jsonl','forbidden.json']}))
    event('selection_complete',buffer_questions=512,probe_groups=len(picked))


def anchors():
    if (OUT/'anchor_manifest.json').exists():
        d=read(OUT/'anchor_manifest.json'); assert sha(OUT/'anchors.jsonl')==d['sha256']; return
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source=ROOT/'experiments/cd_prm_pilot_20261021_gpu/artifacts/public_anchor_source.parquet'
    assert sha(source)==SOURCE_SHA
    forbidden=set(read(OUT/'forbidden.json'));allowed={r['question_sha256'] for r in rows(ROOT/'data/prepared/train.jsonl')}
    best={0:{},1:{}}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=1024):
        for r in batch.to_pylist():
            qh=question_key(r['prompt'])
            if qh in forbidden or qh not in allowed:continue
            assert len(r['completions'])==len(r['labels'])
            for j,label in enumerate(r['labels']):
                label=int(label);steps=r['completions'][:j+1]
                if not steps or not all(s.strip() for s in steps):continue
                ident=hashlib.sha256((r['prompt']+'\0'+'\n\n'.join(steps)).encode()).hexdigest();h=rank('anchor',ident)
                if qh not in best[label] or h<best[label][qh][0]:
                    best[label][qh]=(h,dict(anchor_id=ident,question=r['prompt'],question_sha256=qh,steps=steps,label=label,source_step=j,source_index=r['index']))
    tok=AutoTokenizer.from_pretrained(PRM,local_files_only=True); selected=[]; excluded=collections.Counter()
    for label in [0,1]:
        count=0
        for _,r in sorted(best[label].values(),key=lambda x:x[0]):
            try:e=encode_steps(tok,r['question'],r['steps'])
            except (AssertionError,ValueError):excluded['boundary']+=1;continue
            if len(e['input_ids'])>CFG['anchor_max_tokens']:excluded['length']+=1;continue
            selected.append({**r,**e});count+=1
            if count==1024:break
        assert count==1024, (label,count)
    write_rows(OUT/'anchors.jsonl',selected)
    atomic(OUT/'anchor_manifest.json',dict(source_repository='HuggingFaceH4/prm800k-trl-dedup',source_revision='fd8cba17d8214bfbabf6e7e10f29da4e6614dd6c',source_sha256=SOURCE_SHA,
        sha256=sha(OUT/'anchors.jsonl'),count=len(selected),labels=dict(collections.Counter(r['label'] for r in selected)),excluded=dict(excluded),forbidden_question_overlap=0,
        boundaries='original annotated step list preserved; supervision only at final native marker of each prefix'))
    event('anchors_ready',count=len(selected))


def schedule():
    if (OUT/'data_ready.json').exists():load_data();return
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(PRM,local_files_only=True)
    data={'fail':[],'success':[],'anchor_positive':[],'anchor_negative':[]};excluded=collections.Counter()
    raw=rows(OUT/'buffer_rollouts.jsonl');assert len(raw)==2048
    for r in raw:
        if not valid_for_prm(r):excluded[r['status']]+=1;continue
        try:e=encode_qwen(tok,r['question'],r['completion'],normalize=True)
        except (AssertionError,ValueError):excluded['boundary']+=1;continue
        if len(e['input_ids'])>CFG['prm_max_tokens']:excluded['prm_length']+=1;continue
        data['success' if r['reward'] else 'fail'].append({**r,**e})
    for r in rows(OUT/'anchors.jsonl'):data['anchor_positive' if r['label'] else 'anchor_negative'].append(r)
    assert len(data['anchor_positive'])==len(data['anchor_negative'])==1024
    assert data['fail'] and data['success']
    steps=max(512,math.ceil(len(data['fail'])/2),math.ceil(len(data['success'])/2))
    assert steps<=1024
    rng=random.Random(CFG['seed']);indices={}
    for key,items in data.items():
        seq=[]
        while len(seq)<steps*2:
            ix=list(range(len(items)));rng.shuffle(ix);seq.extend(ix)
        indices[key]=[seq[2*i:2*i+2] for i in range(steps)]
    batches=[{k:indices[k][i] for k in data} for i in range(steps)]
    for key,items in data.items():
        assert set(j for b in batches for j in b[key])==set(range(len(items)))
    atomic(OUT/'encoded_training.json',data)
    atomic(OUT/'batch_manifest.json',dict(steps=steps,batches=batches,counts={k:len(v) for k,v in data.items()},
        input_tokens=sum(len(data[k][j]['input_ids']) for b in batches for k in data for j in b[k]),
        anchors_distinct_used=len({data[k][j]['anchor_id'] for b in batches for k in ['anchor_positive','anchor_negative'] for j in b[k]}),
        encoded_sha256=sha(OUT/'encoded_training.json'),buffer_sha256=sha(OUT/'buffer_rollouts.jsonl'),anchors_sha256=sha(OUT/'anchors.jsonl'),excluded=dict(excluded)))
    atomic(OUT/'data_ready.json',dict(batch_manifest_sha256=sha(OUT/'batch_manifest.json'),policy_updates=0,config_sha256=sha(HERE/'config.json')))
    event('data_ready',steps=steps,counts={k:len(v) for k,v in data.items()})


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['select','schedule']);args=a.parse_args()
    assert_root();verify_release()
    if args.phase=='select':select();anchors()
    else:schedule()
