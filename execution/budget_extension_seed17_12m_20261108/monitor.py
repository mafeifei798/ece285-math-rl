import json,time
from inspect_remote import snapshot,compact,render
def main():
    previous=None
    while True:
        for attempt in range(3):
            try:x=snapshot();break
            except Exception as e:
                print(json.dumps({'read_only_monitor_error':str(e),'attempt':attempt+1}),flush=True)
                if attempt==2:raise
                time.sleep(20)
        render(x);c=compact(x)
        marker=(c['stage'],tuple(((v['step'] or 288)//24,v['smoke_passed'],v['first_update432'],v['first_update576']) for v in c['arms'].values()))
        if marker!=previous:print(json.dumps(c,ensure_ascii=False),flush=True);previous=marker
        if c['stage'] in ['complete','failed'] or not c['alive']:return
        time.sleep(55)
if __name__=='__main__':main()
