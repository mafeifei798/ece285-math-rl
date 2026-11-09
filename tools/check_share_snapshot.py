"""Check a public code snapshot without importing training/model libraries."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import warnings

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERNS = {
    'github_token': r'\bgh[pousr]_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b',
    'huggingface_token': r'\bhf_[A-Za-z0-9]{25,}\b',
    'OpenAI_key': r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b',
    'AWS_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
}


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z'],
                                    cwd=ROOT).decode().split('\0')
    paths = sorted(set(p for p in paths if p))
    syntax = []
    findings = []
    counts = Counter()
    total = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        for rel in paths:
            path = ROOT / rel
            if not path.is_file():
                continue
            total += path.stat().st_size
            if path.suffix in {'.safetensors', '.pt', '.pth', '.bin', '.jsonl', '.zip'}:
                findings.append({'path': rel, 'kind': 'excluded_artifact_extension'})
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            counts['text_files'] += 1
            if path.suffix == '.py':
                counts['python_files'] += 1
                try:
                    ast.parse(text, filename=rel)
                except SyntaxError as exc:
                    syntax.append({'path': rel, 'line': exc.lineno, 'message': exc.msg})
            for name, pattern in TOKEN_PATTERNS.items():
                for match in re.finditer(pattern, text):
                    # Obvious mock/example values are not credentials.
                    tail = match.group().split('_', 1)[-1]
                    if len(set(tail.lower())) <= 1:
                        continue
                    findings.append({'path': rel, 'line': text.count('\n', 0, match.start()) + 1,
                                     'kind': name})
    manifest = json.loads((ROOT / 'SHARE_MANIFEST.json').read_text())
    hashes_checked = 0
    for entry in manifest['copied_files']:
        path = ROOT / entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['shared_sha256'], entry['path']
        hashes_checked += 1
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'snapshot_date': '2026-11-09', 'python_files_syntax_checked': counts['python_files'],
              'text_files_pattern_scanned': counts['text_files'], 'files_considered': len(paths),
              'snapshot_bytes': total, 'source_copy_hashes_checked': hashes_checked,
              'credential_pattern_findings': findings, 'syntax_errors': syntax,
              'new_training_or_model_calls': False,
              'note': 'Static parsing, saved-label checks and pattern scanning only; not end-to-end training reproduction or a guarantee that every possible secret is detected.',
              'passed': not findings and not syntax}
    (ROOT / 'SHARE_CHECKS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
