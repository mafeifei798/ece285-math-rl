"""Pinned public PURE PRM assets. No model loading, sampling, or training."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import requests

REPO = 'jinachris/PURE-PRM-7B'
PIN = 'b0cce6763f2317762c9cb46720c8fe7b07f44a88'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def valid(path, r):
    if not path.exists() or path.stat().st_size != r['size']: return False
    if 'lfs' in r: return digest(path) == r['lfs']['sha256']
    b = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == r['blobId']

def atomic(p, x):
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(x, indent=2)+'\n'); t.replace(p)

def main():
    model, meta, out = map(Path, sys.argv[1:4])
    out.mkdir(parents=True, exist_ok=True); model.mkdir(parents=True, exist_ok=True)
    lock = (out/'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    d = json.loads(meta.read_text()); assert d['id'] == REPO and d['sha'] == PIN
    allowed = {'README.md','config.json','added_tokens.json','merges.txt','special_tokens_map.json',
               'tokenizer_config.json','vocab.json','model.safetensors.index.json'}
    records = [r for r in d['siblings'] if r['rfilename'] in allowed or r['rfilename'].endswith('.safetensors')]
    start = time.monotonic()
    atomic(out/'download_status.json', dict(status='running',pid=os.getpid(),repo=REPO,revision=PIN))
    def fetch(r):
        target = model/r['rfilename']; part = target.with_suffix(target.suffix+'.partial')
        if target.exists():
            assert valid(target,r), 'Existing asset differs; never overwrite: '+str(target)
            return dict(file=r['rfilename'],sha256=digest(target),bytes=r['size'],reused=True)
        url = f'https://hf-mirror.com/{REPO}/resolve/{PIN}/{r["rfilename"]}'
        for attempt in range(4):
            offset = part.stat().st_size if part.exists() else 0
            if offset == r['size'] and valid(part,r): part.replace(target); break
            assert offset < r['size']
            try:
                headers = {'Range':f'bytes={offset}-'} if offset else {}
                with requests.get(url,headers=headers,stream=True,timeout=(20,45)) as response:
                    response.raise_for_status()
                    if offset:
                        assert response.status_code == 206
                        assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    with part.open('ab' if offset else 'wb') as f:
                        for chunk in response.iter_content(4*1024*1024): f.write(chunk)
                assert valid(part,r),'Official digest mismatch'
                part.replace(target); break
            except Exception as exc:
                print(json.dumps(dict(event='retry',file=r['rfilename'],attempt=attempt+1,error=type(exc).__name__)),flush=True)
                if attempt == 3: raise
                time.sleep(2)
        result = dict(file=r['rfilename'],sha256=digest(target),bytes=r['size'],reused=False)
        print(json.dumps(dict(event='verified',**result)),flush=True)
        return result
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: files=list(pool.map(fetch,records))
        atomic(out/'model_assets.json',dict(complete=True,repo=REPO,revision=PIN,files=files,
            metadata_sha256=digest(meta),elapsed_seconds=time.monotonic()-start,
            transport='HF mirror with official pinned blob validation',training=False))
        atomic(out/'download_status.json',dict(status='complete',pid=os.getpid()))
    except Exception as exc:
        atomic(out/'download_status.json',dict(status='failed',pid=os.getpid(),error=type(exc).__name__))
        raise

if __name__ == '__main__': main()
