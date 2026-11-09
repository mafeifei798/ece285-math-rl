"""Finite read-only watcher in the current task. Does not launch or repair jobs."""
import time,json
from inspect_remote import snapshot,compact
from render_status import render


def main():
    previous=None
    while True:
        for attempt in range(3):
            try:x=snapshot();break
            except Exception as e:
                print(json.dumps({'read_only_monitor_error':str(e),'attempt':attempt+1}),flush=True)
                if attempt==2:raise
                time.sleep(20)
        render(x)
        phase=(x['controller'] or {}).get('stage')
        marker=(phase,tuple(((v['progress'] or {}).get('additional_updates',0)//8,bool(v['result']),bool(v['failure'])) for v in x['arms'].values()))
        if marker!=previous:
            print(json.dumps(compact(x),ensure_ascii=False),flush=True);previous=marker
        if phase in ('complete','failed') or not x['controller_alive']:
            return
        time.sleep(55)


if __name__=='__main__':main()
