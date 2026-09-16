"""Read-only packaging checks; no experiment imports or training."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parent.parent


def main():
    manifest = json.loads((ROOT / 'docs/file_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        path = ROOT / item['path']
        # Streaming also works on Python 3.10 and avoids loading participant files.
        raw = path.read_bytes()
        canonical = raw if item.get('hash_mode') == 'binary' else raw.replace(b'\r\n', b'\n')
        assert hashlib.sha256(canonical).hexdigest() == item['sha256'], path
    scripts = list((ROOT / 'src').glob('*.py'))
    for path in scripts:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    figures = sorted((ROOT / 'figures').glob('Fig*.pdf'))
    assert [p.name for p in figures] == [
        'Fig1_framework.pdf', 'Fig2_main_results.pdf',
        'Fig3_hrf_parameters.pdf', 'Fig4_permutation_validation.pdf']
    print(f'PASS: {len(manifest)} file hashes; {len(scripts)} Python syntax checks; four final figures.')
    print('This does not import optional dependencies or rerun experiments.')


if __name__ == '__main__':
    main()
