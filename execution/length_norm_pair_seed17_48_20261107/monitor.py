"""Follow the single already-dispatched finite controller, read-only remotely."""
import json,time
from inspect_remote import snapshot,compact


if __name__=='__main__':
    previous=None;failures=0
    while True:
        try:
            x=snapshot();failures=0;c=compact(x)
            key=(c['controller_stage'],tuple((a,(v['updates'] or 0)//8,v['finished']) for a,v in c['arms'].items()),
                 (c['evaluation'] or {}).get('done',0)//256)
            if key!=previous:
                print(json.dumps(c,ensure_ascii=False),flush=True);previous=key
            stage=c['controller_stage']
            if stage=='complete':break
            if stage=='failed':raise SystemExit(1)
            if not c['controller_alive']:raise RuntimeError('Recorded controller no longer alive; inspect without redispatch')
        except Exception as exc:
            failures+=1;print(json.dumps({'monitor_error':str(exc),'consecutive_failures':failures}),flush=True)
            if failures>=3:raise
        time.sleep(30)
