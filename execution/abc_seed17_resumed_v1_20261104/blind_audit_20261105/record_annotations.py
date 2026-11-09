"""Freeze assistant annotations made from X/Y packets before opening the model key."""
from pathlib import Path
import json,hashlib,time
P=Path(__file__).resolve().parent
# expected_outcome applies the frozen last-box + completed-output protocol,
# not semantic correctness of unboxed prose. None denotes uncertainty.
data=[
('R01','X',1,False,'invalid','The asserted equality 7(sqrt(5)-1)/2 = 14/3 is false. The presented quadratic does not support the boxed answer. Figure is absent from supplied problem; geometry cannot be fully checked.'),
('R01','Y',0,False,'uncertain_geometry','14(sqrt(2)-1) is not the reference answer. Its geometric assumptions cannot be verified without the missing figure.'),
('R02','X',0,False,'invalid','57 needed, 48 available; needs9, not48-8=40. Confuses available chairs with required chairs.'),
('R02','Y',1,True,'valid_body_answer_conflict','57-(8+5*8)=9 correctly derived; leading box105 conflicts with correct final box9.'),
('R03','X',0,True,'correct_core_unfinished','A42, E30, difference12. Repeated misleading qualifications and opening37; ends in repetition without completed answer.'),
('R03','Y',1,True,'valid_body_answer_conflict','Correct chain30→35→42 and difference12. Invents a reference37 and gives conflicting boxes, but ends12.'),
('R04','X',0,False,'invalid','Halves52 instead of the26 female performers, yielding29 rather than16.'),
('R04','Y',1,True,'valid_body_answer_conflict','52/2/2+3=16. Opening26 is incorrect, final16 matches derivation.'),
('R05','X',0,True,'valid_unboxed','50*11/4=137.5; correct prose conclusion1371/2 but lacks required answer box.'),
('R05','Y',1,True,'valid','Correct conversion11/4, product275/2 and mixed1371/2, boxed.'),
('R06','X',0,False,'invalid_repetition','Derives tan2 correctly but repeatedly insists1/2 and ends mid-sentence.'),
('R06','Y',1,True,'valid','Divide sinA=2cosA by nonzero cosA to obtain tanA=2.'),
('R07','X',1,True,'valid','31/11111=279/99999=0.overline(00279), period5; prime-length block is not a repetition of one digit.'),
('R07','Y',0,False,'invalid','11111=41*271 is not prime. Even for prime denominator p-1 is only a possible maximal period, not a guaranteed period.'),
('R08','X',1,True,'valid','10+2*10=30, boxed.'),
('R08','Y',0,True,'valid_unboxed','Correct10+20=30, but only empty opening box and unboxed conclusion.'),
('R09','X',1,True,'valid','Two paid scoops cost3 and give3 total;6 buys2 bundles=6 scoops. Distinguishes4 paid from6 received.'),
('R09','Y',0,False,'invalid_final_override','Derives6 total, then final box4 without justification. Under benchmark total-received reading, final wrong; paid-scoop wording is noted.'),
('R10','X',1,True,'valid','Reflect center(0,0) to(8,2), radius5; coefficients1,-16,-4,43. Initial symbolic box is a placeholder.'),
('R10','Y',0,False,'invalid_final_override','Correct reflected circle in body then invents given coefficients and overrides with(1,10,-24,40).'),
('R11','X',0,True,'valid_core_imprecise_labels_unboxed','10000-2000-5000-1000=2000; text mistakenly calls8000 steps before jog instead of after, but intended subtraction and prose answer correct; no valid box.'),
('R11','Y',1,True,'self_corrected_imprecise_labels','Initially says4000 then explicitly corrects to8000-6000=2000. Still misnames8000 as before jog; final2000 valid. Not a clean error-free derivation.'),
('R12','X',1,True,'valid','First4+2, second2+8, total16.'),
('R12','Y',0,False,'invalid_repetition','Correct total16 repeatedly overridden by invented reference18; opening18 and unfinished repetition.'),
('R13','X',0,True,'valid_unboxed','Marcus26-5=21; Leo2*21=42. Empty opening box, correct prose final42.'),
('R13','Y',1,True,'valid_body_answer_conflict','Correct42 derived; invents supplied52 and contradicts opening52, but final42.'),
('R14','X',1,True,'valid','Using benchmark72-foot start,72*(2/3)^2=32.'),
('R14','Y',0,False,'invalid','Computes32 second bounce then incorrectly applies third decay to21.33; opening16 also contradictory.'),
('R15','X',0,True,'valid_unboxed','2^8=2^(2x) gives4, correctly in prose/code. Only empty box; no valid answer box.'),
('R15','Y',1,True,'valid','2x=8 gives boxed4.'),
('R16','X',1,False,'invalid_case_exclusion','Incorrectly excludes omega1 as zero denominator; denominators actually2. All roots still give1, but completeness argument is invalid.'),
('R16','Y',0,False,'invalid','Same false exclusion of omega1; derives1 and even reports numeric1, then asserts1/2.'),
('R17','X',0,True,'valid_unboxed','20-5*2=10, correct prose answer; empty box only.'),
('R17','Y',1,True,'valid','20-10=10, boxed.'),
('R18','X',1,False,'invalid_incomplete_proof','Wrong sign in y derivative and erroneous cross term in x simplification. Restricting to x=-y and axes does not prove a global bound, despite correct final1/4. Opening1/12 conflicts.'),
('R18','Y',0,False,'invalid','Wrong derivative/algebra, then fabricates output1/12 contradicting its own f(1,-1)=1/4 lower bound.'),
('R19','X',1,True,'valid_body_answer_conflict','7*10*(21/30)=49 correct; initial7/9 conflicts with final49.'),
('R19','Y',0,False,'invalid_repetition','Calculates49 then invents required2 and repeats without completed correct box.'),
('R20','X',1,True,'valid','(4*6*1) mod7=3, correct boxed answer.'),
('R20','Y',0,False,'invalid_final_override','Derives3 but gives final0, unsupported override.')]
annotations=[dict(case_id=i,output=o,expected_outcome=e,correct_math_core=valid,verdict=v,reason=why) for i,o,e,valid,v,why in data]
assert len(annotations)==40 and len({(x['case_id'],x['output']) for x in annotations})==40
p=P/'annotations_blinded.json';assert not p.exists()
p.write_text(json.dumps({'reviewer':'assistant, model identity and stored outcome labels blinded; not human panel','annotations':annotations},ensure_ascii=False,indent=2)+'\n')
(P/'annotation_freeze.json').write_text(json.dumps({'annotations_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'blind_packet_sha256':hashlib.sha256((P/'blind.json').read_bytes()).hexdigest(),'unix':time.time(),'before_key_read':True},indent=2)+'\n')
print('40 response annotations frozen. Model key has not been opened by this script.')
