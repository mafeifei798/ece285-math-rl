"""Pin the already-chosen question unit metadata and code before inference."""
from pathlib import Path
import json
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/AB_prompt_validation_20261022'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'training'))
from common import atomic_json,file_hash,read_jsonl
sys.path.insert(0,str(ROOT/'experiments/reward_repair_v3_20261021'))
from repairs import make_metadata

PERCENT_REASONS={
 'math/train/counting_and_probability/321':'asks smallest possible percent of adults',
 'gsm8k/train/515':'asks percentage chance of serious accident',
 'gsm8k/train/5695':'asks percentage of chips',
 'gsm8k/train/2472':'explicitly asks probability expressed as a percentage',
 'gsm8k/train/4014':'asks percentage of bottle liquid',
 'gsm8k/train/6830':'asks percentage of gingerbread men',
 'gsm8k/train/2299':'asks percentage of games',
 'gsm8k/train/7446':'asks percent of school population',
 'gsm8k/train/4820':'explicitly asks amount expressed as percentage of a bottle',
 'gsm8k/train/2204':'asks percent discount',
}
POINT_REASONS={'gsm8k/train/1635':'difference of grade percentages; absolute percentage-point difference'}

def main():
    if (OUT/'release.json').exists():
        rel=json.loads((OUT/'release.json').read_text())
        for p,h in rel['input_sha256'].items():assert file_hash(ROOT/p)==h
        print('Existing frozen release verified, not changed.');return
    assert not (OUT/'predictions.jsonl').exists()
    rows=read_jsonl(OUT/'questions.jsonl');by_id={r['problem_id']:r for r in rows}
    original=json.loads((ROOT/'audits/reward_repair_v3_20261021/unit_metadata.json').read_text())
    units=dict(original);decisions=[]
    for r in json.loads((OUT/'unit_candidates.json').read_text()):
        pid=r['problem_id'];kind='legacy';reason='Percent appears in inputs or code, but requested answer has another unit; preserve original fallback.'
        if pid in PERCENT_REASONS:kind='percent_value';reason=PERCENT_REASONS[pid]
        if pid in POINT_REASONS:kind='percentage_points';reason=POINT_REASONS[pid]
        if pid=='gsm8k/train/7274':reason='Asks odds without explicit output unit; conservatively retain legacy grading, no inferred percentage conversion.'
        if kind!='legacy':
            assert pid not in units
            units[pid]=make_metadata(r['problem'],r['answer'],kind)
        decisions.append({'problem_id':pid,'answer_type':kind,'reason':reason})
    assert all(units[k]==v for k,v in original.items())
    assert set(PERCENT_REASONS)|set(POINT_REASONS)<=set(by_id)
    atomic_json(OUT/'unit_metadata.json',units)
    atomic_json(OUT/'unit_decisions.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'reviewed_before_any_completions':True,'all_existing_metadata_unchanged':True,
        'source':'Question and gold schema only; no policy outputs observed.',
        'new_percent_value':len(PERCENT_REASONS),'new_percentage_points':len(POINT_REASONS),'decisions':decisions,
        'rule_changed':False,'note':'Metadata extends the existing frozen unit-aware rule to new questions; historical outputs and metadata are untouched.'})
    inputs={str(p.relative_to(ROOT)):file_hash(p) for p in HERE.glob('*.py')}
    for name in ['protocol.json','questions.jsonl','dataset_audit.json','unit_candidates.json','unit_metadata.json','unit_decisions.json','protection_manifest.json','blind_review_question_ids.json']:
        p=OUT/name;inputs[str(p.relative_to(ROOT))]=file_hash(p)
    old_release=json.loads((ROOT/'experiments/ab_corrected_6m_seed17_20261022/release.json').read_text())
    for rel in ['training/baseline_config.json','receipts/assets.json','training/common.py',
        'experiments/reward_repair_v3_20261021/repairs.py','experiments/reward_repair_v2_20261021/v2core.py',
        'experiments/reward_repair_v2_20261021/legacy_snapshot/source_manifest.json',
        'experiments/reward_repair_v2_20261021/legacy_snapshot/training/common.py',
        'experiments/reward_repair_v2_20261021/legacy_snapshot/training/reward.py']:
        inputs[rel]=file_hash(ROOT/rel)
    for arm in 'AB':
        for name in ['adapter_model.safetensors','adapter_config.json']:
            rel=f'runs/{arm}_corrected_seed17_6M_20261022/final_adapter/{name}'
            inputs[rel]=file_hash(ROOT/rel)
    release={'created_utc':datetime.now(timezone.utc).isoformat(),'frozen_before_generation':True,
        'no_new_training':True,'input_sha256':inputs,
        'old_pair_release_sha256':file_hash(ROOT/'experiments/ab_corrected_6m_seed17_20261022/release.json'),
        'verifier_source_matches_old_pair':file_hash(ROOT/'experiments/reward_repair_v3_20261021/repairs.py')=='df8ac0198652a81a90e70979a6b48b5111dc15ba19edc041411d776657020da3'}
    assert release['verifier_source_matches_old_pair']
    atomic_json(OUT/'release.json',release)
    print(json.dumps({'frozen':True,'new_unit_metadata_entries':11,'input_files':len(inputs),'release_sha256':file_hash(OUT/'release.json')},indent=2))

if __name__=='__main__':main()
