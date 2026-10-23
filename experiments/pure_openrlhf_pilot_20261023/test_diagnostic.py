"""Small independent CPU fixtures; no ML dependency or experiment execution."""
import copy
import json
import math
import unittest
from types import SimpleNamespace

from diagnostic import summarize_batch


def fixture_row(text,ids,terminated,positions,*,processed=None,outcome=0,pid='q'):
    actor=list(processed if processed is not None else ids)
    return dict(problem_id=pid,emitted_completion=text,emitted_completion_ids=list(ids),
        original_terminated=terminated,outcome=outcome,alignment=dict(
            actor_completion_ids=actor,actor_boundary_positions=list(positions),
            step_count=len(positions),process_sequences_applied=True,
            emitted_ended_in_eos=terminated,process_sequences_changed=actor!=ids,
            terminal_replaced_with_eos=not terminated))


class DiagnosticTests(unittest.TestCase):
    def test_categories_raw_vs_processed_and_no_mutation(self):
        rows=[fixture_row('',[99],True,[0]),
              fixture_row(' \r\n\t ',[1,2,99],True,[2]),
              fixture_row('Thank you.',[3,4,99],True,[2]),
              fixture_row('x=2.\r\n\r\n\\boxed{2}',[5,6,7],False,[2],processed=[5,6,99],outcome=1),
              fixture_row('<special>',[8,99],True,[1])]
        before=copy.deepcopy(rows)
        result=summarize_batch(rows)
        self.assertEqual(rows,before)
        self.assertEqual(result['counts']['empty'],2)
        self.assertEqual(result['counts']['whitespace_only'],1)
        self.assertEqual(result['counts']['natural_eos_only'],1)
        self.assertEqual(result['counts']['short_nonempty'],3)
        self.assertEqual(result['counts']['no_box'],4)
        self.assertEqual(result['counts']['no_sep'],5)
        self.assertEqual(result['per_response'][3]['raw_content_tokens'],3)
        self.assertEqual(result['per_response'][3]['processed_content_tokens'],2)
        self.assertTrue(result['per_response'][3]['truncated'])
        self.assertFalse(result['per_response'][4]['empty'])
        self.assertIsNone(result['advantages'])
        self.assertIsNone(result['process_scores'])
        json.dumps(result,allow_nan=False)

    def test_short_threshold_counts_content_not_terminal_eos(self):
        rows=[fixture_row('a',[1]*32+[99],True,[32]),
              fixture_row('b',[1]*33+[99],True,[33]),
              fixture_row('c',[1]*33,False,[32],processed=[1]*32+[99])]
        result=summarize_batch(rows)
        self.assertEqual(result['counts']['short'],1)
        self.assertEqual([r['raw_content_tokens'] for r in result['per_response']],[32,33,33])
        self.assertEqual(result['lengths']['raw_content_tokens']['median'],33.)

    def test_advantage_mass_uses_valid_mask_and_overlapping_categories(self):
        rows=[fixture_row('',[99],True,[0]),fixture_row('ok',[1,2,99],True,[2],outcome=1),
              fixture_row('long',[1]*40+[99],True,[40],pid='other')]
        # Valid masks are deliberately short synthetic arrays; this tests
        # aggregation independently of token alignment validation upstream.
        batch=SimpleNamespace(arm='P-PRM',mask=[[1,0,0],[1,1,1],[1,1,0]],
            advantages=[[2,100,100],[-1,3,0],[5,-1,100]],
            boundary=[[1,0,0],[0,0,1],[0,1,0]],task=[[.8,999,999],[0,0,.2],[0,.1,999]])
        before=copy.deepcopy(batch.__dict__)
        result=summarize_batch(rows,batch=batch)
        self.assertEqual(batch.__dict__,before)
        adv=result['advantages'];self.assertEqual(adv['positive_tokens'],3)
        self.assertEqual(adv['positive_advantage_mass'],10.)
        self.assertEqual(adv['valid_tokens'],6)
        self.assertEqual(adv['positive_boundary_tokens'],1)
        self.assertAlmostEqual(adv['categories']['empty']['share_of_all_positive_advantage_mass'],.2)
        self.assertAlmostEqual(adv['categories']['short']['share_of_all_positive_advantage_mass'],.5)
        self.assertEqual(adv['categories']['short_nonempty']['positive_tokens'],1)
        self.assertAlmostEqual(result['task_rewards']['response_task_reward_sum']['mean'],1.1/3)

    def test_zero_positive_advantage_has_null_share_not_nan(self):
        rows=[fixture_row('',[99],True,[0])]
        batch=dict(mask=[[1]],advantages=[[0]],boundary=[[1]],task=[[1]],arm='P-PRM')
        result=summarize_batch(rows,batch=batch)
        self.assertIsNone(result['advantages']['categories']['empty']['share_of_all_positive_advantage_mass'])
        self.assertEqual(result['advantages']['positive_tokens'],0)
        json.dumps(result,allow_nan=False)

    def test_native_vs_fp32_scores_remain_separate(self):
        rows=[fixture_row('x\n\ny',[1,2,99],True,[0,2])]
        scored=[dict(native_step_scores=[.5,.75],native_signed_scores=[0,.5],
                     step_scores=[.5001,.7501],signed_scores=[.0002,.5002])]
        before=copy.deepcopy(scored)
        result=summarize_batch(rows,scored=scored)
        self.assertEqual(scored,before)
        self.assertEqual(result['counts']['no_sep'],0)
        self.assertEqual(result['process_scores']['native_step_scores']['response_min']['mean'],.5)
        self.assertEqual(result['process_scores']['step_scores']['response_min']['mean'],.5001)
        self.assertIsNone(result['task_rewards'])

    def test_missing_eos_evidence_stays_unknown(self):
        rows=[dict(emitted_completion='x',emitted_completion_ids=[1,2],original_terminated=True,
                   outcome=None,alignment={})]
        result=summarize_batch(rows)
        self.assertEqual(result['known_denominators']['short'],0)
        self.assertIsNone(result['per_response'][0]['raw_content_tokens'])
        self.assertEqual(result['outcomes']['missing'],1)
        self.assertIsNone(result['outcomes']['accuracy'])

    def test_shape_mismatch_and_gpu_inputs_fail(self):
        rows=[fixture_row('',[99],True,[0])]
        with self.assertRaises(ValueError):
            summarize_batch(rows,scored=[])
        with self.assertRaises(ValueError):
            summarize_batch(rows,batch=dict(mask=[[1]],advantages=[[1,2]]))
        class GPUArray:
            device=SimpleNamespace(type='cuda')
            def tolist(self): raise AssertionError('Must not access GPU data')
        with self.assertRaises(ValueError):
            summarize_batch(rows,batch=dict(mask=GPUArray(),advantages=[[1]]))

    def test_empty_batch_and_nonfinite_guard(self):
        result=summarize_batch([])
        self.assertEqual(result['rows'],0)
        self.assertIsNone(result['rates']['empty'])
        json.dumps(result,allow_nan=False)
        with self.assertRaises(ValueError):
            summarize_batch([fixture_row('',[99],True,[0])],
                batch=dict(mask=[[1]],advantages=[[math.nan]]))


if __name__=='__main__':
    unittest.main()
