"""Codex per-record mathematical review, recorded separately from PRM scores.

The reviewer inspected question, gold and full steps first. These annotations
are diagnostic opinions by one assistant, not independent human gold labels.
An early false assertion followed by a valid correction is explicitly separated
from a wholly valid proof and from an unresolved wrong derivation.
"""
from collections import Counter
import json
from pathlib import Path
from prepare import OUT, read, save

ANNOTATIONS = [
    ("B01", "error", 1, 2, False,
     "Less than 3 has two die outcomes, not three; P=(2/6)(3/6)=1/6. Both the opening 1/9 and later 1/4 are wrong. Later logarithm/matrix/finance questions are unrelated.",
     {1:"false_answer_claim",2:"mathematical_error",3:"off_topic",4:"off_topic",5:"off_topic",6:"off_topic"}, []),
    ("B02", "recovered_error", 3, 3, True,
     "Step3 says the x difference is5 and distance5.16. The actual x difference is12. Steps4-15 then correctly compute sqrt(12^2+5^2)=13. Correct final answer does not make the earlier claim true.",
     {1:"format_only",2:"format_only",3:"mathematical_error",**{i:"valid_correction" for i in range(4,16)}}, []),
    ("B03", "error", 1, 3, False,
     "(2x-3)^2=4x^2-12x+9, so a=9. Opening a=16 or0 is false. Completing-square notation at step2 can be valid for a suitable f, but step3 confuses coefficients and solves an unrelated equation incorrectly. No final boxed answer.",
     {1:"false_answer_claim",2:"conditional_identity",3:"mathematical_error"}, []),
    ("B04", "error", 9, 9, False,
     "For x>30 the inner absolute is2x-60, so the outer RHS is60 and x=60 is a third solution. For x<=30 the roots are12 and20. The sum is92, not32. Steps2-6 are unrelated/ill-formed rather than a valid prefix.",
     {1:"format_only",2:"uninterpretable",3:"irrelevant",4:"irrelevant",5:"unsupported",6:"unsupported",7:"plan",8:"plan",9:"mathematical_error",10:"missing_solution",11:"mathematical_error",19:"repeated_error",20:"repeated_error",21:"mathematical_error",24:"invalid_conclusion"}, []),
    ("B05", "error", 1, 1, False,
     "At a=1, x^2-x+1 does not have a root at1 (value1; discriminant-3). Integer coefficients do not require integer roots. F(1)=92, not0. Polynomial division by x^2-x+2 has zero remainder, verifying gold2.",
     {1:"mathematical_error",2:"valid_formula_in_wrong_setup",3:"unsupported_integer_root_assumption",4:"dependent_on_error",5:"mathematical_error",6:"conditional_algebra",7:"dependent_on_error",8:"invalid_conclusion"}, []),
    ("B06", "valid", None, None, True,
     "The discriminant is110.25 and its square root10.5. Roots are-5/7 and10/7; selecting the positive root is correct. All22 raw segments were inspected; the opening descriptive box is superseded by the correct final box.",
     {i:"valid" for i in range(1,23)}, list(range(1,23))),
    ("B07", "recovered_error", 1, 1, True,
     "Opening boxed26 is false. The subsequent computation is valid: g(5)=8, f(8)=53, f(5)=20, g(20)=23, difference30, correctly boxed at the end.",
     {1:"false_answer_claim",**{i:"valid_correction" for i in range(2,8)}}, []),
    ("B08", "recovered_error", 1, 1, True,
     "With height4 and area30, base=2*30/4=15, not15/2. Later steps correct the result to(-15,0), but the only boxed value remains the wrong-15/2. This is a stale-box output failure as well as an earlier mathematical error.",
     {1:"mathematical_error",2:"correction_plan",3:"valid_correction",4:"valid_correction_unboxed"}, []),
    ("B09", "target_solved_then_contaminated", 3, 3, True,
     "The pizza calculation and boxed31/32 at step1 are valid. It then generates another problem and boxes210. That extra problem actually needs540-340=200, not210. The last-box protocol therefore extracts an unrelated wrong answer.",
     {1:"valid_target_solution",2:"off_topic",3:"mathematical_error_and_answer_contamination"}, [1]),
    ("B10", "error", 4, 4, False,
     "Steps2-3 preserve the rational expression. Step4's simplification to(y+1)^2/y is false (take x=2,y=1/2). At y=1 the original denominator vanishes. Since A=B^2+2, A/B=B+2/B>=2sqrt2 for B>0.",
     {1:"format_only",2:"valid",3:"valid",4:"mathematical_error",5:"mathematical_error"}, [2,3]),
    ("B11", "error", 1, 7, False,
     "The opening195.2 is wrong. Steps2-6 correctly obtain n=3 and mean10. Step7 then unjustifiably adds n a second time and changes the answer to13.",
     {1:"false_answer_claim",2:"valid_correction",3:"plan",4:"valid_formula",5:"substitution",6:"valid_target_solution",7:"mathematical_error",8:"dependent_on_error",9:"invalid_conclusion"}, []),
    ("B12", "valid", None, None, True,
     "The proposed loop covers i=5,...,29 and math.ceil(sqrt(i)) is correct for these small integers. Independent computation gives112, matching the final box. All7 segments were inspected.",
     {i:"valid" for i in range(1,8)}, list(range(1,8))),
    ("B13", "incomplete_with_error", 9, 11, True,
     "Steps6-8 correctly solve continuity: 2a+4=5, a=1/2. The output then switches to another piecewise function and invents a horizontal-intersection requirement. Step11 simultaneously asserts f(2)=1 and0. It loops and truncates without a final boxed target answer.",
     {1:"format_only",2:"off_topic",3:"off_topic",4:"valid_general_rule",5:"plan",6:"valid_target_derivation",7:"plan",8:"valid_target_solution",9:"wrong_problem_substitution",10:"unsupported_constraint",11:"mathematical_error",13:"dependent_on_error",25:"fabricated_or_inconsistent_execution_output",28:"mathematical_error",40:"fabricated_or_inconsistent_execution_output",43:"incomplete_error"}, []),
    ("B14", "answer_correct_process_wrong", 5, 5, True,
     "Gold8 is correct: odd squares are1 mod8 and m=3,n=1 yields8. But step5 case15 has m-n=2, not-2, and36 is4 mod8, not0. Step6 also falsely says80 is not divisible by16;80=5*16. The proof is not made valid by the final8.",
     {1:"correct_answer_claim",2:"valid",3:"valid",4:"plan",5:"mathematical_error",6:"mathematical_error",7:"correct_answer_with_flawed_proof"}, [1,2,3,4]),
    ("B15", "recovered_error", 1, 1, True,
     "Opening79 ways is false. The three-state recurrence, initial values(1,1,0), loop and eventual504 are correct. An independent enumeration of all1024 binary delivery patterns gives504.",
     {1:"false_answer_claim",**{i:"valid_correction" for i in range(2,9)}}, []),
    ("B16", "target_solved_then_incomplete_contamination", 16, 20, True,
     "Steps1-10 correctly obtain midpoint(-2,4) and product-8, including the correct box. Later unrelated questions are generated. Step16 restricts real variables to a small integer search without justification; step20 confuses d with1/d. The response eventually truncates. No PRM score was requested for this whole response.",
     {**{i:"valid_target_solution" for i in range(1,11)},**{i:"off_topic" for i in range(11,16)},16:"unsupported_search_restriction",20:"mathematical_error",21:"inconsistent_execution_output",22:"invalid_unrelated_answer",**{i:"off_topic" for i in range(23,37)}}, list(range(1,11))),
    ("B17", "incomplete_with_error", 1, 5, False,
     "The congruences give18 and53 below60, sum71. Opening15/126 are wrong. Step5's code would return([18,53],71), not the claimed([1,38],39); these are fabricated execution results. It later drifts to unrelated problems and truncates.",
     {1:"false_answer_claim",2:"false_answer_claim",3:"question_repetition",4:"garbled_prefix_then_valid_setup",5:"fabricated_execution_output",6:"mathematical_error",**{i:"off_topic" for i in range(7,29)}}, []),
    ("B18", "valid_missing_answer_box", None, None, True,
     "87=3*29,89 is prime,91=7*13,93=3*31. The six distinct primes and plain-text answer6 are correct. The only box is empty, so the output is invalid under the unchanged extraction protocol; it never received a PRM score.",
     {1:"format_only",2:"valid",3:"valid_target_solution_unboxed"}, [1,2,3]),
    ("B19", "recovered_error_with_stale_box", 1, 11, True,
     "Opening7 is wrong. Steps2-10 correctly derive sqrt59. Step11 invents a requirement that the question says boxed7; step13 boxes8, although no rounding was requested. The final prose returns to sqrt59, leaving the last boxed answer wrong.",
     {1:"false_answer_claim",**{i:"valid_correction" for i in range(2,11)},11:"invented_question_constraint",12:"unsupported_rounding",13:"wrong_boxed_answer",14:"valid_unboxed_correction"}, []),
    ("B20", "valid_percent_verifier_false_negative", None, None, True,
     "Independence gives0.60*0.25=0.15=15%. The question explicitly requests a percentage and boxed15% is mathematically correct. Gold stores15 in percentage units. Treating15% as0.15 while treating gold15 as an untyped scalar creates a false negative in the legacy outcome verifier.",
     {i:"valid" for i in range(1,6)}, list(range(1,6))),
]


def main():
    records = read(OUT / "selected_records.json")
    annotations = []
    for data, item in zip(records, ANNOTATIONS):
        ident, status, first_error, first_bad_deduction, target_solved, reasoning, overrides, valid_prefix = item
        assert data["sample_id"] == ident
        n = len(data["steps"])
        assert first_error is None or 1 <= first_error <= n
        steps = [{"step":i,"judgment":overrides.get(i,"dependent_or_not_independently_adjudicated"),
                  "comment":"See record-level mathematical verification; later dependent steps are not treated as independent evidence."}
                 for i in range(1,n+1)]
        ann = {"sample_id":ident,"source_row":data["source_row"],"math_status":status,
            "first_error_step":first_error,"first_bad_deduction_step":first_bad_deduction,
            "has_correct_target_solution_or_conclusion":target_solved,"mathematical_reason":reasoning,
            "valid_prefix_step_indices":valid_prefix,"steps":steps,
            "reviewer":"Codex single-reviewer mathematical inspection; not independent human gold",
            "blinding":"Question, gold and complete step text inspected before PRM score review. The target gold was visible; outcome reward and PRM scores were hidden in blind_review.txt."}
        annotations.append(ann)
    save(OUT / "mathematical_annotations.json", annotations)
    print("saved_blind_annotations",len(annotations))


if __name__=="__main__":
    main()
