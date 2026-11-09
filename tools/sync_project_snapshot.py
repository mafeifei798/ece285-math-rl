"""Refresh this review repository from the original local experiment directory.

Only copies source, small reports/configs and aggregate evidence. No model code
is imported or run. Existing historical repository files are retained.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ('training', 'training_b', 'operations', 'experiments', 'execution',
           'evidence', 'proposal_neurips', 'midterm_neurips_20261109')
ROOT_FILES = ('project_plan.html', 'project_plan_simple.html', 'protocol.json',
              'protocol_simple.json', 'recompute_evidence.py')
CODE = {'.py', '.sh', '.js', '.ts'}
TEXT = CODE | {'.json', '.txt', '.md', '.yaml', '.yml', '.toml', '.cfg', '.ini',
               '.html', '.css', '.tex', '.bib', '.sty', '.csv', '.svg'}
IMAGES = {'.png', '.jpg', '.jpeg'}
EXCLUDED_DIRS = {'.git', '__pycache__', '.venv', 'venv',
                 'checkpoints', 'final_adapter', 'cache', 'plot_cache',
                 'raw_generation', 'batches', 'paired_batches', 'snapshots',
                 'previous_draft', 'github_sync_20261109'}
FULL_TEXT_KEYS = {'completion', 'full_completion', 'response_text', 'raw_response',
                  'question', 'prompt', 'problem', 'solution', 'reference_solution'}
IP = re.compile(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def contains_generation(value):
    if isinstance(value, dict):
        if any(k in FULL_TEXT_KEYS and isinstance(v, str) and len(v) > 120
               for k, v in value.items()):
            return True
        return any(contains_generation(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_generation(v) for v in value)
    return False


def sanitize(text):
    text = re.sub(r"/Users/[^/\s\"'\\]+/Desktop/python/ece285_math_rl_project",
                  '/path/to/ece285_math_rl_project', text)
    text = re.sub(r"/(?:yimin|scratch|mnt|workspace)/(?:[^/\s\"'\\]+/)+ece285_math_rl_project",
                  '/path/to/ece285_math_rl_project', text)
    text = re.sub(r"/yimin/[^/\s\"'\\]+/[^/\s\"'\\]+", '/path/to/project_parent', text)
    text = re.sub(r"/Users/[^/\s\"'\\]+", '/home/USER', text)
    text = re.sub(r"/home/[^/\s\"'\\]+", '/home/USER', text)
    def redact_ip(match):
        value = match.group()
        parts = [int(x) for x in value.split('.')]
        if all(p <= 255 for p in parts) and parts[0] >= 10 and value not in {
                '127.0.0.1', '255.255.255.255'} and parts[:3] != [192, 0, 2]:
            return 'GPU_HOST'
        return value
    text = IP.sub(redact_ip, text)
    return re.sub(r'[A-Za-z_][A-Za-z0-9_-]*@GPU_HOST', 'USER@GPU_HOST', text)


def candidates(source):
    for folder in FOLDERS:
        for directory, dirs, names in os.walk(source / folder):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for name in names:
                path = Path(directory) / name
                if not path.is_symlink():
                    yield path
    for name in ROOT_FILES:
        path = source / name
        if path.is_file():
            yield path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    source = args.source.resolve()
    assert source != ROOT and (source / 'training/baseline_config.json').is_file()
    copied = []
    excluded = Counter()
    changed = Counter()
    syntax_errors = []
    for path in sorted(candidates(source)):
        rel = path.relative_to(source)
        ext = path.suffix.lower()
        if ext in IMAGES and any(tag in path.stem.lower() for tag in
                                 ('resource_probe', 'public_repository', 'screenshot')):
            excluded['non_scientific_screenshot'] += 1
            continue
        allowed = ext in TEXT or path.name in {'LICENSE', 'NOTICE', 'COPYING'}
        allowed |= ext == '.pdf' and rel.parts[0] == 'midterm_neurips_20261109'
        allowed |= ext in IMAGES and ('figures' in rel.parts or len(rel.parts) <= 3)
        if not allowed:
            excluded['dataset_binary_archive_or_transient_extension'] += 1
            continue
        if path.stat().st_size > 2_000_000:
            excluded['large_file'] += 1
            continue
        raw = path.read_bytes()
        data = raw
        if ext in TEXT or path.name in {'LICENSE', 'NOTICE', 'COPYING'}:
            try:
                text = raw.decode('utf-8')
            except UnicodeDecodeError:
                excluded['non_utf8_text'] += 1
                continue
            if ext == '.json':
                value = json.loads(text)
                if contains_generation(value):
                    excluded['full_prompt_or_generation_JSON'] += 1
                    continue
            if ext == '.csv' and any(key in text.splitlines()[0].lower().split(',')
                                     for key in FULL_TEXT_KEYS):
                excluded['full_prompt_or_generation_CSV'] += 1
                continue
            text = sanitize(text)
            if ext == '.json':
                json.loads(text)
            if ext == '.py':
                try:
                    ast.parse(text, filename=str(rel))
                except SyntaxError as exc:
                    syntax_errors.append({'path': str(rel), 'line': exc.lineno, 'message': exc.msg})
            data = text.encode('utf-8')
        dest = ROOT / rel
        is_changed = not dest.exists() or dest.read_bytes() != data
        if is_changed:
            changed['new' if not dest.exists() else 'updated'] += 1
            if args.apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
        copied.append({'path': str(rel), 'original_sha256': sha(raw),
                       'shared_sha256': sha(data), 'shared_bytes': len(data),
                       'redacted': data != raw})
    if syntax_errors:
        raise RuntimeError(json.dumps(syntax_errors))
    manifest = {'snapshot_date': '2026-11-09', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'visibility': 'public', 'mode': 'overlay_without_deleting_historical_files',
                'scope': 'Source, small reports/configs, aggregate evidence and midterm PDF; no GPU operation',
                'source_root': 'original experiment directory, kept local',
                'copied_file_count': len(copied), 'copied_bytes': sum(e['shared_bytes'] for e in copied),
                'redactions': ['personal local paths', 'server project roots', 'server address and SSH target'],
                'original_hashes_are_pre_redaction': True,
                'old_releases_and_receipts_refer_to_original_experiments': True,
                'excluded_counts': dict(excluded), 'copied_files': copied}
    if args.apply:
        (ROOT / 'SHARE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'apply': args.apply, 'files': len(copied), 'bytes': manifest['copied_bytes'],
                      'changes': dict(changed), 'excluded': dict(excluded),
                      'python_syntax_errors': syntax_errors}))


if __name__ == '__main__':
    main()
