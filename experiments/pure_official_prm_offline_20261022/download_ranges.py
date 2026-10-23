"""Complete public weight downloads via bounded HTTP ranges, with official SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import requests


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,obj):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(obj,indent=2)+'\n');p.replace(path)


def main():
    model,metadata,out=map(Path,sys.argv[1:4])
    info=json.loads(metadata.read_text())
    assert info['id']=='jinachris/PURE-PRM-7B' and info['sha']=='b0cce6763f2317762c9cb46720c8fe7b07f44a88'
    statepath=out/'download_ranges_progress.json'
    if statepath.exists():raise RuntimeError('Inspect existing range download before resuming')
    import fcntl
    process_lock=(out/'download.lock').open('a');fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=[r for r in info['siblings'] if r['rfilename'] not in ('.gitattributes','training_args.bin','trainer_state.json')];files=[];tasks=[];fds={};state={};start=time.monotonic();lock=threading.Lock()
    for record in records:
        name=record['rfilename']
        if name=='.gitattributes':continue
        target=model/name
        if not name.endswith('.safetensors'):
            assert target.stat().st_size==record['size']
            raw=target.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==record['blobId']
            files.append({'file':name,'sha256':digest(target),'bytes':len(raw),'reused':True});continue
        if target.exists():
            assert target.stat().st_size==record['size'] and digest(target)==record['lfs']['sha256']
            files.append({'file':name,'sha256':record['lfs']['sha256'],'bytes':record['size'],'reused':True});continue
        partial=target.with_suffix(target.suffix+'.partial')
        prefix=partial.stat().st_size if partial.exists() else 0
        assert prefix<=record['size']
        fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o664);fds[name]=fd
        state[name]={'prefix_bytes_reused':prefix,'total_bytes':record['size'],'completed_ranges':[]}
        for lo in range(prefix,record['size'],32*1024*1024):
            hi=min(lo+32*1024*1024,record['size'])-1
            tasks.append((name,lo,hi,record['size']))
    def publish():
        obj={'complete':False,'elapsed_seconds':time.monotonic()-start,'threads':12,'files':state,
             'verified_complete_file_bytes':sum(f['bytes'] for f in files),
             'downloaded_prefix_and_completed_range_bytes':sum(v['prefix_bytes_reused']+sum(z['bytes'] for z in v['completed_ranges']) for v in state.values())}
        atomic(statepath,obj)
    publish()
    def fetch(task):
        name,lo,hi,total=task
        url=f"https://hf-mirror.com/jinachris/PURE-PRM-7B/resolve/{info['sha']}/{name}?range_start={lo}"
        for attempt in range(4):
            try:
                with requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(10,60)) as r:
                    r.raise_for_status()
                    assert r.status_code==206,'Range transport did not honor byte range'
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
                    assert match and tuple(map(int,match.groups()))==(lo,hi,total),'Cached/misaligned HTTP range'
                    written=0;h=hashlib.sha256()
                    for block in r.iter_content(1024*1024):
                        assert written+len(block)<=hi-lo+1
                        count=os.pwrite(fds[name],block,lo+written)
                        assert count==len(block);written+=count;h.update(block)
                    assert written==hi-lo+1
                os.fsync(fds[name])
                with lock:
                    state[name]['completed_ranges'].append({'start':lo,'end':hi,'bytes':written,'sha256':h.hexdigest()})
                    publish()
                return
            except Exception as e:
                if attempt==3:raise
                print(json.dumps({'stage':'range_retry','file':name,'start':lo,'exception':type(e).__name__,'attempt':attempt+1}),flush=True)
                time.sleep(2*(attempt+1))
    # Interleave files so one slower object cannot monopolize all connections.
    tasks.sort(key=lambda t:(t[1],t[0]))
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed([pool.submit(fetch,t) for t in tasks]):future.result()
    for record in records:
        name=record['rfilename']
        if name not in state:continue
        os.close(fds[name]);partial=model/(name+'.partial');target=model/name
        assert partial.stat().st_size==record['size']
        actual=digest(partial);assert actual==record['lfs']['sha256'],name+' differs from official HF blob'
        partial.replace(target)
        files.append({'file':name,'sha256':actual,'bytes':record['size'],'reused_prefix_bytes':state[name]['prefix_bytes_reused'],
                      'transport':'HF mirror bounded HTTP ranges, official pinned HF SHA256 verified'})
        print(json.dumps({'stage':'official_weight_hash_verified','file':name,'bytes':record['size']}),flush=True)
    result={'complete':True,'repo':info['id'],'revision':info['sha'],'files':files,
            'metadata_sha256':digest(metadata),'elapsed_seconds':time.monotonic()-start,
            'training':False,'model_updates':0}
    atomic(out/'model_assets.json',result)
    current=json.loads(statepath.read_text());current['complete']=True;current['elapsed_seconds']=time.monotonic()-start
    atomic(statepath,current)
    print(json.dumps({'complete':True,'seconds':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
