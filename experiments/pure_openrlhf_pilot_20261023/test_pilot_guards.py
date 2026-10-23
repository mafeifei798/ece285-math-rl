"""Behavioral CPU tests for the pilot's preregistered two-window stop guard.

Imports the real guard without constructing PilotEngine or touching CUDA.
Fixtures represent complete 32-response batches, including valid short answers.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from train_arm import collapse_check


def response(kind):
    base=dict(emitted_completion='A valid longer derivation. \\boxed{2}',
              emitted_completion_ids=[1]*40+[99],raw_ended_in_eos=True,
              reliably_parsed=True,outcome=1,original_terminated=True)
    if kind=='correct_short':
        return dict(base,emitted_completion='\\boxed{2}',emitted_completion_ids=[1,2,99])
    if kind=='wrong_parsed_short':
        return dict(base,emitted_completion='\\boxed{3}',emitted_completion_ids=[1,3,99],outcome=0)
    if kind=='empty':
        return dict(base,emitted_completion='',emitted_completion_ids=[99],reliably_parsed=False,outcome=0)
    if kind=='unparsed_short':
        return dict(base,emitted_completion='Thank you.',emitted_completion_ids=[1,2,99],reliably_parsed=False,outcome=0)
    if kind=='content32_plus_eos':
        return dict(base,emitted_completion='Unparseable but nonempty.',emitted_completion_ids=[1]*32+[99],reliably_parsed=False,outcome=0)
    if kind=='content33_truncated':
        return dict(base,emitted_completion='Unparseable and incomplete.',emitted_completion_ids=[1]*33,
                    raw_ended_in_eos=False,original_terminated=False,reliably_parsed=False,outcome=0)
    return base


def write_window(root,start,kind,count):
    (root/'batches').mkdir(exist_ok=True)
    for step in range(start,start+5):
        rows=[response(kind) for _ in range(count)]+[response('normal') for _ in range(32-count)]
        (root/'batches'/f'{step:04d}.json').write_text(json.dumps(dict(rows=rows)))


class PilotGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()

    def test_correct_short_answers_never_count_as_collapse(self):
        write_window(self.root,1,'correct_short',32);write_window(self.root,6,'correct_short',32)
        stopped,report=collapse_check(self.root,10)
        self.assertFalse(stopped)
        self.assertTrue(all(w['short_unparsed_rate']==0 and w['empty_rate']==0 for w in report['windows']))

    def test_wrong_but_parsed_short_answers_are_not_unparsed(self):
        write_window(self.root,1,'wrong_parsed_short',32);write_window(self.root,6,'wrong_parsed_short',32)
        self.assertFalse(collapse_check(self.root,10)[0])

    def test_exact_empty_threshold_requires_both_windows(self):
        write_window(self.root,1,'empty',8);write_window(self.root,6,'empty',7)
        self.assertFalse(collapse_check(self.root,10)[0])
        write_window(self.root,6,'empty',8)
        stopped,report=collapse_check(self.root,10)
        self.assertTrue(stopped)
        self.assertEqual([w['n'] for w in report['windows']],[160,160])
        self.assertEqual([w['empty_rate'] for w in report['windows']],[.25,.25])
        self.assertEqual(len(report['reasons']),1)

    def test_both_windows_just_below_empty_threshold_do_not_stop(self):
        write_window(self.root,1,'empty',7);write_window(self.root,6,'empty',7)
        self.assertFalse(collapse_check(self.root,10)[0])

    def test_exact_nonempty_unparsed_short_threshold(self):
        write_window(self.root,1,'unparsed_short',16);write_window(self.root,6,'unparsed_short',16)
        stopped,report=collapse_check(self.root,10)
        self.assertTrue(stopped)
        self.assertEqual([w['empty_rate'] for w in report['windows']],[0.,0.])
        self.assertEqual([w['short_unparsed_rate'] for w in report['windows']],[.5,.5])

    def test_different_failure_types_do_not_fake_persistence_of_one_trigger(self):
        write_window(self.root,1,'empty',8);write_window(self.root,6,'unparsed_short',16)
        self.assertFalse(collapse_check(self.root,10)[0])

    def test_short_threshold_excludes_only_real_terminal_eos(self):
        write_window(self.root,1,'content32_plus_eos',16);write_window(self.root,6,'content32_plus_eos',16)
        self.assertTrue(collapse_check(self.root,10)[0])
        write_window(self.root,1,'content33_truncated',16);write_window(self.root,6,'content33_truncated',16)
        self.assertFalse(collapse_check(self.root,10)[0])

    def test_only_adjacent_last_ten_batches_and_no_file_mutation(self):
        write_window(self.root,1,'empty',32);write_window(self.root,6,'empty',32)
        write_window(self.root,11,'correct_short',32);write_window(self.root,16,'correct_short',32)
        files=list((self.root/'batches').glob('*.json'))
        before={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        self.assertFalse(collapse_check(self.root,20)[0])
        self.assertEqual(before,{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files})

    def test_before_ten_batches_no_stop_or_file_access(self):
        self.assertEqual(collapse_check(self.root,9),(False,[]))


if __name__=='__main__':
    unittest.main()
