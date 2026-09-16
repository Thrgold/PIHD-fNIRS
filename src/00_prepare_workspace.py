"""Restore the historical results layout without rerunning experiments.

Run once after cloning. Archived result files remain unchanged. Existing runtime
files are never overwritten: remove/rename them yourself if restoring is needed.
Only Python standard-library dependencies are required.
"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent.parent


def digest(path, mode):
    raw = path.read_bytes()
    canonical = raw if mode == 'binary' else raw.replace(b'\r\n', b'\n')
    return hashlib.sha256(canonical).hexdigest()


def main():
    entries = json.loads((ROOT / 'docs/file_manifest.json').read_text(encoding='utf-8'))
    copied = 0
    conflicts = []
    for entry in entries:
        if entry['kind'] != 'archived_result':
            continue
        source = ROOT / entry['path']
        if digest(source, entry.get('hash_mode')) != entry['sha256']:
            raise ValueError(f'Archived file differs from manifest: {source}')
        destination = ROOT / entry['source_relative']
        if destination.exists():
            if digest(destination, entry.get('hash_mode')) != entry['sha256']:
                conflicts.append(str(destination.relative_to(ROOT)))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    for folder in ['figures', 'output/pdf', 'results/npy', 'results/json']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f'Restored {copied} files into results/. No training was run.')
    if conflicts:
        print('Existing different files were preserved:', *conflicts, sep='\n')
    if not (ROOT / 'data/eeg_features_all_v6.npy').exists():
        print('Data not installed: see data/README.md before training.')


if __name__ == '__main__':
    main()
