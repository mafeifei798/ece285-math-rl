"""Resume verified byte ranges with persistent HTTP sessions and pinned digests."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import requests
from fetch_assets import valid, digest, atomic, PIN, REPO

def main():
    model,metadata,out,initial,urls_path=map(Path,sys.argv[1:6])
    lockfile=(out/'download.lock').open('a');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=json.loads(metadata.read_text());assert info['id']==REPO and info['sha']==PIN
    prior=json.loads(initial.read_text());urls=json.loads(urls_path.read_text())
    state=json.loads(json.dumps(prior['files']));files=[];fds={};tasks=[]
    start=time.monotonic();lock=threading.Lock();sessions=threading.local()
    progress=out/'download_tail_progress.json'
    for r in info['siblings']:
        name=r['rfilename']
        if name in ('.gitattributes','trainer_state.json','training_args.bin'):continue
        target=model/name
        if target.exists():
            assert valid(target,r);files.append(dict(file=name,sha256=digest(target),bytes=r['size'],reused=True));continue
        assert name.endswith('.safetensors') and urls.get(name)
        part=model/(name+'.partial');assert part.exists()
        s=state[name];assert s['total_bytes']==r['size']
        fd=os.open(part,os.O_RDWR);fds[name]=fd
        completed=set()
        for block in s['completed_ranges']:
            data=os.pread(fd,block['bytes'],block['start'])
            assert len(data)==block['bytes'] and hashlib.sha256(data).hexdigest()==block['sha256']
            completed.add((block['start'],block['end']))
        for lo in range(s['prefix_bytes_reused'],r['size'],32*1024*1024):
            hi=min(lo+32*1024*1024,r['size'])-1
            if (lo,hi) not in completed:
                for sublo in range(lo,hi+1,4*1024*1024):
                    subhi=min(sublo+4*1024*1024,hi+1)-1
                    if (sublo,subhi) not in completed:tasks.append((name,sublo,subhi,r['size']))
    def publish(status='running',errors=None):
        atomic(progress,dict(complete=False,status=status,pid=os.getpid(),elapsed_seconds=time.monotonic()-start,
            files=state,remaining_ranges_initial=len(tasks),threads=8,errors=errors or [],
            downloaded_prefix_and_completed_range_bytes=sum(v['prefix_bytes_reused']+sum(z['bytes'] for z in v['completed_ranges']) for v in state.values())))
    publish()
    def fetch(task):
        name,lo,hi,total=task
        if not hasattr(sessions,'session'):sessions.session=requests.Session()
        for attempt in range(10):
            try:
                request_start=time.monotonic()
                with sessions.session.get(urls[name],headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(10,20)) as response:
                    response.raise_for_status();assert response.status_code==206
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    assert m and tuple(map(int,m.groups()))==(lo,hi,total)
                    n=0;h=hashlib.sha256()
                    for b in response.iter_content(256*1024):
                        if time.monotonic()-request_start>45:raise TimeoutError("Slow range transfer; retain verified blocks and retry")
                        assert n+len(b)<=hi-lo+1;assert os.pwrite(fds[name],b,lo+n)==len(b);h.update(b);n+=len(b)
                    assert n==hi-lo+1
                os.fsync(fds[name])
                with lock:
                    state[name]['completed_ranges'].append(dict(start=lo,end=hi,bytes=n,sha256=h.hexdigest()));publish()
                return None
            except Exception as exc:
                print(json.dumps(dict(event='range_retry',file=name,start=lo,attempt=attempt+1,error=type(exc).__name__)),flush=True)
                sessions.session.close();sessions.session=requests.Session()
                if attempt<9:time.sleep(min(10,attempt+1))
        return dict(file=name,start=lo,end=hi,error='retry budget exhausted')
    errors=[];tasks.sort(key=lambda t:(t[1],t[0]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch,t) for t in tasks]):
            error=future.result()
            if error:errors.append(error)
    if errors:publish('failed',errors);raise RuntimeError('Unfinished ranges retained; resume from this progress file')
    for r in info['siblings']:
        name=r['rfilename']
        if name not in fds:continue
        os.close(fds[name]);part=model/(name+'.partial');assert valid(part,r),name+' official SHA mismatch'
        part.replace(model/name)
        files.append(dict(file=name,sha256=r['lfs']['sha256'],bytes=r['size'],reused_prefix_bytes=state[name]['prefix_bytes_reused']))
        print(json.dumps(dict(event='official_weight_hash_verified',file=name)),flush=True)
    atomic(out/'model_assets.json',dict(complete=True,repo=REPO,revision=PIN,files=files,metadata_sha256=digest(metadata),
        elapsed_seconds=time.monotonic()-start,transport='cached public mirror CDN URLs, persistent HTTP range sessions; official hashes verified',training=False,model_updates=0))
    publish('complete');d=json.loads(progress.read_text());d['complete']=True;atomic(progress,d)

if __name__=='__main__':main()
