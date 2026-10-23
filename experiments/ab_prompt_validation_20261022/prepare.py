"""Freeze held-out questions and two prompts before observing any completions."""
from pathlib import Path
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/AB_prompt_validation_20261022'
sys.path.insert(0, str(ROOT/'training'))
from common import atomic_json, file_hash, question_key, read_jsonl, write_jsonl

ORIGINAL = 'Question: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:'
FINAL_ONLY = 'Question: {problem}\n\nPlease reason step by step before giving the answer. Do not put an answer in a box while reasoning. At the end, state your final answer exactly once within \\boxed{{}}, consistent with your completed reasoning. Do not add any text after that final answer.\nAnswer:'
ADAPTER_HASHES = {'A':'3016479cf0dca6e0b627551f027e20b706d0c80d1a2574191b4f1386542fd862',
                  'B':'c98d03b3a4d67223cf36552cbd1c11bbcf17148b15843fbd1b26b373abfa957f'}

def main():
    if (OUT/'dataset_audit.json').exists():
        old=json.loads((OUT/'dataset_audit.json').read_text())
        assert file_hash(OUT/'questions.jsonl')==old['questions_sha256']
        assert file_hash(OUT/'protocol.json')==old['protocol_sha256']
        print('Existing preparation verified; no re-sampling.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    assert not list(OUT.glob('*predictions*')), 'Outputs already exist'
    cfg=json.loads((ROOT/'training/baseline_config.json').read_text())
    manifest=json.loads((ROOT/'data/prepared/manifest.json').read_text())
    assets=json.loads((ROOT/'receipts/assets.json').read_text())
    assert cfg['prompt_template']==ORIGINAL
    data={name:read_jsonl(ROOT/'data/prepared'/f'{name}.jsonl') for name in ['train','validation','math500','gsm8k']}
    for name,rows in data.items():
        assert file_hash(ROOT/'data/prepared'/f'{name}.jsonl')==manifest['files'][name+'.jsonl']['sha256']
        assert all(question_key(r['problem'])==r['question_sha256'] for r in rows)
    forbidden={r['question_sha256'] for name in ['train','math500','gsm8k'] for r in data[name]}
    pb=json.loads((ROOT/'data/raw/processbench_math.json').read_text())
    forbidden.update(question_key(r['problem']) for r in pb)
    # These two validation rows were used only in an earlier 8-token LoRA interface check.
    excluded_smoke=data['validation'][:2]
    math_rows=[dict(r,dataset='math_validation') for r in data['validation'][2:]]
    assert len(math_rows)==498
    assert not {r['question_sha256'] for r in math_rows}&forbidden
    from datasets import Dataset
    arrow=ROOT/'cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow'
    ds=Dataset.from_file(str(arrow))
    assert len(ds)==7473
    gsm=[];seen=set();excluded=[]
    for i,r in enumerate(ds):
        key=question_key(r['question'])
        if key in forbidden or key in seen or key in {m['question_sha256'] for m in math_rows}:
            excluded.append(i);continue
        seen.add(key)
        assert '####' in r['answer']
        gsm.append({'problem_id':f'gsm8k/train/{i}','problem':r['question'],
            'answer':r['answer'].rsplit('####',1)[1].strip().replace(',',''),
            'question_sha256':key,'dataset':'gsm8k_validation'})
    random.Random(285).shuffle(gsm)
    gsm=gsm[:500]
    rows=math_rows+gsm
    assert len({r['problem_id'] for r in rows})==998
    assert len({r['question_sha256'] for r in rows})==998
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(assets['model_path'],local_files_only=True)
    for r in rows:
        for name,template in [('original',ORIGINAL),('final_only',FINAL_ONLY)]:
            r['prompt_'+name]=template.format(problem=r['problem'])
            r['prompt_tokens_'+name]=len(tokenizer.encode(r['prompt_'+name],add_special_tokens=False))
            assert r['prompt_tokens_'+name]<=2048, (r['problem_id'],name)
        r.pop('prompt',None);r.pop('prompt_tokens',None)
    candidate_keys={r['problem_id'] for r in rows}
    actual_training_overlaps={}
    for arm in 'AB':
        run=ROOT/f'runs/{arm}_corrected_seed17_6M_20261022'
        ids={r['problem_id'] for r in read_jsonl(run/'rollouts.jsonl')}
        actual_training_overlaps[arm]=sorted(ids&candidate_keys)
        assert not actual_training_overlaps[arm]
        assert file_hash(run/'final_adapter/adapter_model.safetensors')==ADAPTER_HASHES[arm]
    protected={}
    for arm in 'AB':
        for f in sorted((ROOT/f'runs/{arm}_corrected_seed17_6M_20261022').rglob('*')):
            if f.is_file():protected[str(f.relative_to(ROOT))]={'bytes':f.stat().st_size,'sha256':file_hash(f)}
    for folder in ['training','experiments/ab_corrected_6m_seed17_20261022','experiments/reward_repair_v3_20261021','experiments/reward_repair_v2_20261021']:
        for f in sorted((ROOT/folder).rglob('*')):
            if f.is_file() and '__pycache__' not in str(f):protected[str(f.relative_to(ROOT))]={'bytes':f.stat().st_size,'sha256':file_hash(f)}
    for rel in ['audits/reward_repair_v3_20261021/unit_metadata.json','data/prepared/manifest.json','receipts/assets.json']+[f'data/prepared/{n}.jsonl' for n in data]:
        f=ROOT/rel;protected[rel]={'bytes':f.stat().st_size,'sha256':file_hash(f)}
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'experiment':'ab_prompt_validation_20261022',
        'authorization':'User requested existing A/B checkpoint prompt comparison on independent validation data; inference only; original outputs and grading preserved.',
        'design':'2x2: A/B adapters x original/final_only prompts; same questions, greedy settings and frozen verifier.',
        'templates':{'original':ORIGINAL,'final_only':FINAL_ONLY},'adapter_sha256':ADAPTER_HASHES,
        'datasets':{'math_validation':498,'gsm8k_validation':500},'cells':['A_original','A_final_only','B_original','B_final_only'],
        'generation':{'temperature':0.0,'top_p':1.0,'top_k':-1,'max_tokens':2048,'seed':285,'max_model_len':4096,
            'max_num_seqs':32,'max_num_batched_tokens':4096,'gpu_memory_utilization':0.25,'dtype':'bfloat16','enforce_eager':True,'max_loras':2},
        'question_chunk_size':16,'cell_order':'Rotate order each question chunk to balance execution order; same fixed order in receipts.',
        'max_completions':3992,'max_generated_completion_tokens':8175616,
        'training':False,'prm_scoring':False,'checkpoint_selection':False,'prompt_search':False,
        'primary_metrics':['Frozen official accuracy per dataset and cell','Paired B-A under each prompt; paired prompt effect per adapter; interaction (Bfinal-Boriginal)-(Afinal-Aoriginal)'],
        'secondary_metrics':['Truncated and invalid-box counts','Exactly one final box with no substantive text after it','Early last box: ends within first25% characters and >=100 characters follow','Assistant blind qualitative consistency review of fixed random12 questions x4 cells'],
        'semantic_review':{'GSM_questions':8,'MATH_questions':4,'selection_seed':2851701,'blind_to_arm_prompt_reward_and_gold_first':True,'not_a_new_grader':True},
        'analysis':'10000 paired question bootstrap draws seed285; retain all four cells in each draw. Report uncertainty, no post-hoc accuracy relabeling.',
        'unit_metadata':'Apply frozen repaired_outcome implementation; new validation question units declared before generation from question/gold only. Old metadata entries unchanged.',
        'limitations':['Question-hash and ID disjointness from current RL training and original test sets; paraphrase/pretraining exposure not ruled out.',
            'All498 MATH validation questions appear in the known PRM800K question inventory; not a PRM-clean benchmark. No PRM is used in this inference comparison.',
            'This isolates inference prompt sensitivity on fixed policies, not causal training-time PRM effects.',
            'Small blind consistency sample is qualitative; full-set box-position compliance is not proof of mathematical consistency.']}
    write_jsonl(OUT/'questions.jsonl',rows)
    atomic_json(OUT/'protocol.json',protocol)
    atomic_json(OUT/'protection_manifest.json',protected)
    units=json.loads((ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json').read_text())
    candidates=[{k:r[k] for k in ['problem_id','dataset','problem','answer']} for r in rows if re.search(r'percent|%',r['problem'],re.I)]
    atomic_json(OUT/'unit_candidates.json',candidates)
    review=[]
    rng=random.Random(protocol['semantic_review']['selection_seed'])
    for dataset,n in [('math_validation',4),('gsm8k_validation',8)]:
        review+=rng.sample([r['problem_id'] for r in rows if r['dataset']==dataset],n)
    atomic_json(OUT/'blind_review_question_ids.json',review)
    audit={'prepared':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'questions_sha256':file_hash(OUT/'questions.jsonl'),'protocol_sha256':file_hash(OUT/'protocol.json'),
        'source_validation_sha256':manifest['files']['validation.jsonl']['sha256'],'source_gsm_train_arrow_sha256':file_hash(arrow),
        'gsm_revision':'740312add88f781978c0658806c59bc2815b9866','gsm_original_split':'train','gsm_eligible_count':len(seen),
        'gsm_excluded_duplicate_or_overlap_indices':excluded,'gsm_shuffle_seed':285,
        'excluded_prior_smoke_ids':[r['problem_id'] for r in excluded_smoke],
        'actual_A_B_training_id_overlap':actual_training_overlaps,'train_and_test_normalized_question_overlap':0,
        'unique_questions':len(rows),'unit_candidates':len(candidates),'protected_files':len(protected),
        'prompt_max_tokens':{name:max(r['prompt_tokens_'+name] for r in rows) for name in ['original','final_only']},
        'old_unit_metadata_sha256':file_hash(ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json')}
    atomic_json(OUT/'dataset_audit.json',audit)
    print(json.dumps(audit,indent=2))
    print('UNIT_CANDIDATES',json.dumps(candidates,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
