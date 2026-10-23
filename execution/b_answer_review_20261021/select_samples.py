"""Fixed 30 random + 20 diagnostic B-6M answers; retain an unscored review packet."""
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

OUT=Path(__file__).resolve().parent
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
key=lambda t:hashlib.sha256(re.sub(r'\s+','',unicodedata.normalize('NFKC',t)).encode()).hexdigest()


def main():
    assert not (OUT/'selection.json').exists(),'Keep fixed selection; do not resample after review'
    rows=read(OUT/'source/rollouts.jsonl')
    train={r['problem_id']:r for r in read(OUT/'source/train.jsonl')}
    solutions={key(r['problem']):r for r in read(OUT/'source/prm800k_train_questions.jsonl')}
    eligible=[i for i,r in enumerate(rows) if r['valid_for_process_reward'] and r['status'] in {'exact','symbolic'} and r['terminated']]
    rng=random.Random(20261021)
    random_ix=rng.sample(eligible,30)
    used=set(random_ix)
    chosen=[('R'+str(k+1).zfill(2),i,'uniform_random') for k,i in enumerate(random_ix)]
    pools={
      'answer_correct_min_le_005':[i for i in eligible if rows[i]['outcome_reward']==1 and rows[i]['min_score']<=.05],
      'answer_wrong_min_ge_080':[i for i in eligible if rows[i]['outcome_reward']==0 and rows[i]['min_score']>=.8],
      'single_native_segment_long_text':[i for i in eligible if rows[i]['step_count']==1 and len(rows[i]['completion'])>=300],
    }
    counts={'answer_correct_min_le_005':8,'answer_wrong_min_ge_080':8,'single_native_segment_long_text':4}
    # Already discussed illustrative errors are not deliberately reused as new diagnostic examples.
    old_examples={78,2095,4975}
    for category,n in counts.items():
        pool=[i for i in pools[category] if i not in used and i not in old_examples]
        selected=rng.sample(pool,n)
        for i in selected:
            chosen.append(('S'+str(len(chosen)-29).zfill(2),i,category));used.add(i)
    allrecords=[];blind=[]
    for identifier,i,category in chosen:
        r=rows[i];q=train[r['problem_id']]
        steps=[s.strip() for s in r['completion'].split('\n\n') if s.strip()]
        assert len(steps)==r['step_count']==len(r['step_scores'])
        assert min(r['step_scores'])==r['min_score']
        sol=solutions.get(key(q['problem']))
        allrecords.append({'review_id':identifier,'source_row':i,'sampling_category':category,'question':q['problem'],
          'question_sha256':q['question_sha256'],'steps':steps,'standard_solution':sol['solution'] if sol else None,'saved':r})
        blind.append({'review_id':identifier,'source_row':i,'question':q['problem'],'steps':steps})
    manifest={'seed':20261021,'source_rows':len(rows),'eligible_complete_parseable_rows':len(eligible),
      'excluded':dict(collections.Counter(r['status'] for i,r in enumerate(rows) if i not in set(eligible))),
      'random_n':30,'targeted_n':20,'targeted_rule':counts,'targeted_eligible_pool_sizes':{k:len(v) for k,v in pools.items()},
      'random_draw':'Uniform answer-level sampling without replacement among complete, parseable, PRM-scored outputs; no length/score/subject filter',
      'same_question_can_recur':True,'sampling_is_not_50_iid_answers':True,
      'prior_known_error_rows_not_forced_into_targeted_set':sorted(old_examples),
      'sources':{p.name:sha(p) for p in (OUT/'source').glob('*.jsonl')},
      'selection':[{'review_id':a,'source_row':b,'sampling_category':c} for a,b,c in chosen],
      'reviewer':'Codex reasoning audit, not independent human annotation; uncertain cases are retained',
      'blind_packet':'Scores, final outcome, gold answer, reference solution and targeted subclass hidden in the first review packet'}
    for name,data in [('selection.json',manifest),('selected_full.json',allrecords),('blind_packet.json',blind)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    chunks=[]
    for r in blind:
        chunks.append(f"=== {r['review_id']} [row {r['source_row']}] ===\nQUESTION: {r['question']}\n"+'\n'.join(f"[{i+1}] {s}" for i,s in enumerate(r['steps'])))
    (OUT/'blind_packet.txt').write_text('\n\n'.join(chunks)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['selection','sources']},ensure_ascii=False))
    print('characters',len('\n'.join(chunks)),'steps',sum(len(r['steps']) for r in blind))


if __name__=='__main__':main()
