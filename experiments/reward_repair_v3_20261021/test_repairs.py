"""Real regression cases and mathematical bounds, no model training."""
import json
from pathlib import Path
import unittest
from repairs import *


class Tests(unittest.TestCase):
    def judge(self, answer, gold="15", kind="percent_value", question="What percentage?"):
        return repaired_outcome("\\boxed{" + answer + "}", gold, True, question=question,
            problem_id="q", unit_metadata={"q":make_metadata(question,gold,kind)})

    def test_actual_false_negative_and_non_equivalent_decimals(self):
        for answer in ["15", "15%", "15\\%"]: self.assertEqual(self.judge(answer)["reward"],1)
        for answer in ["0.15", "0.15%", "1500%", "-15%"]: self.assertEqual(self.judge(answer)["reward"],0)

    def test_probability_and_percent_mentions_keep_their_units(self):
        self.assertEqual(self.judge("15\\%","0.15","dimensionless","What is the probability?")["reward"],1)
        self.assertEqual(self.judge("15","0.15","dimensionless","What is the probability?")["reward"],0)
        self.assertEqual(self.judge("700%","700","legacy","Eleven percent of what number is seventy-seven?")["reward"],0)

    def test_question_hash_binding_and_invalid_outputs(self):
        with self.assertRaises(ValueError):
            repaired_outcome("\\boxed{15%}","15",True,question="changed",problem_id="q",
                unit_metadata={"q":make_metadata("original","15","percent_value")})
        for text,term in [("15%",True),("\\boxed{15%}",False)]:
            r=repaired_outcome(text,"15",term,question="Q",problem_id="q",unit_metadata={"q":make_metadata("Q","15","percent_value")})
            self.assertFalse(valid_for_prm(r));self.assertEqual(mixed_reward(r,[]),0)

    def test_actual_B20_and_B04_canonical_roundtrip(self):
        raw="  First\r\n\r\nSecond\r\nline\n\nThird  "
        v=canonical_view(raw,True)
        self.assertEqual(v["steps"],["First","Second\nline","Third"])
        for step,s in zip(v["steps"],v["spans"]):
            self.assertEqual(raw[s["raw_start"]:s["raw_end"]].replace("\r\n","\n"),step)
        for raw in ["x\n\ny", "single step", "a\nb", "", "  "]:
            self.assertEqual(canonical_view(raw,True)["steps"],canonical_view(raw,False)["steps"])

    def test_new_unit_status_gets_same_process_reward(self):
        self.assertAlmostEqual(mixed_reward(self.judge("15%"),[.8,.6]),1.12)

    def test_PRM_adapter_keeps_actor_text_and_maps_scores(self):
        class Fake:
            def score(self,questions,completions):
                self.seen=list(completions)
                return [{"step_count":2,"step_scores":[.9,.8]}],{}
        base=Fake();raw="a\r\n\r\nb"
        result,_=CanonicalPRMAdapter(base).score(["Q"],[raw])
        self.assertEqual(base.seen,["a\n\nb"]);self.assertEqual(raw,"a\r\n\r\nb")
        self.assertEqual(len(result[0]["step_raw_spans"]),2)

    def test_proposed_guardrail_is_bounded_and_preserves_mixed_outcome_signs(self):
        import torch
        for count in range(9):
            y=[1.]*count+[0.]*(8-count)
            for p in [[0.]*8,[1.]*8,[1.]+[0.]*7,[0.]+[1.]*7,list(torch.linspace(0,1,8).tolist())]:
                a,ap=proposed_bounded_advantages(y,p)
                self.assertLessEqual(float(ap.abs().max()),.175001)
                if 0<count<8:
                    self.assertTrue((a[:count]>0).all());self.assertTrue((a[count:]<0).all())
        a,_=proposed_bounded_advantages([1.]*8,[0.]*7+[.001])
        self.assertLess(float(a.abs().max()),.001)


if __name__=="__main__": unittest.main()
