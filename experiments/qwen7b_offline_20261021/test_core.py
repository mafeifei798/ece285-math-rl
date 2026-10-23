"""Checks for serialization, label alignment and actual installed GRPO algebra."""
import json
from pathlib import Path
import sys
import unittest
from core import (auc, canonical_view, cpu_advantages, encode_qwen, midranks,
                  atomic_json)


class TokenizerFixture:
    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            if text.startswith("<extra_0>"):
                result.append(999999); text = text[9:]
            else:
                result.append(ord(text[0])); text = text[1:]
        return result

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize and not add_generation_prompt
        return ''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)


class Tests(unittest.TestCase):
    def test_marker_count_and_positions(self):
        t = TokenizerFixture()
        e = encode_qwen(t, "Question", " first\n\nsecond ")
        self.assertEqual(e['steps'], ['first','second'])
        self.assertEqual([e['input_ids'][p] for p in e['positions']], [999999,999999])
        self.assertLess(e['positions'][-1],len(e['input_ids'])-1)

    def test_literal_marker_rejected(self):
        with self.assertRaises(ValueError):encode_qwen(TokenizerFixture(),'q','x<extra_0>')

    def test_empty_steps_rejected(self):
        with self.assertRaises(ValueError):encode_qwen(TokenizerFixture(),'q',' \n\n ')

    def test_raw_versus_normalized_crlf(self):
        raw = ' a\r\n\r\nb \n\n c '
        self.assertEqual(canonical_view(raw,False)['steps'],['a\r\n\r\nb','c'])
        v=canonical_view(raw,True)
        self.assertEqual(v['steps'],['a','b','c'])
        self.assertEqual([raw[s['raw_start']:s['raw_end']] for s in v['spans']],['a','b','c'])

    def test_tied_midranks(self):
        self.assertEqual(midranks([1.,1.,0.,3.]),[2.5,2.5,4.,1.])

    def test_auc_direction_and_ties(self):
        self.assertEqual(auc([1,2],[0,1]),.875)
        self.assertIsNone(auc([],[0]))

    def test_uniform_rewards(self):
        self.assertEqual(cpu_advantages([1.]*8),[0.]*8)

    def test_actual_trl_against_independent_reference(self):
        import torch,trl
        from trl_kernel import actual_trl_advantages
        source=Path(trl.__file__).parent/'trainer/grpo_trainer.py'
        cases=[[0.,0.,0.,1.,1.,0.,0.,0.], [0.2000,0.2001,0.1999,.2,.2,.1998,.2002,.2], [1.]*8]
        for case in cases:
            _,a=actual_trl_advantages(case,source)
            ref=cpu_advantages(case)
            self.assertLess(max(abs(x-y) for x,y in zip(a.tolist(),ref)),.0003)


if __name__ == '__main__':
    output=Path(sys.argv[1])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    atomic_json(output,{'passed':result.wasSuccessful(),'tests':result.testsRun,
                       'failures':len(result.failures),'errors':len(result.errors),'training':False})
    if not result.wasSuccessful():raise SystemExit(1)
