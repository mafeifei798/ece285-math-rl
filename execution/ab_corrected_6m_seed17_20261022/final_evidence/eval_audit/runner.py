import json,pathlib,subprocess,sys,time
r=pathlib.Path(sys.argv[1]);d=r/'audits/ab_corrected_postprocessing_20261022';o=d/'eval_verification_20261022T1418Z'
start=time.time()
with (o/'stdout_stderr.log').open('x') as f:
 result=subprocess.run([sys.executable,'-B',str(d/'verify_completed_evaluations.py'),'--root',str(r),'--output',str(o/'verified.json')],cwd=r,stdout=f,stderr=subprocess.STDOUT)
receipt={'returncode':result.returncode,'started_unix':start,'finished_unix':time.time(),'output_exists':(o/'verified.json').exists()}
with (o/'process_result.json').open('x') as f:json.dump(receipt,f,indent=2)
