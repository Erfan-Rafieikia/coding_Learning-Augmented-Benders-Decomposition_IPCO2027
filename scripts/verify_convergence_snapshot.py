#!/usr/bin/env python3
"""Freeze/verify the code, configs and input hashes for a fresh pilot."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot_source(repository, root):
    repository, root = Path(repository), Path(root)
    if subprocess.run(['git', '-C', str(repository), 'diff', '--quiet', 'HEAD']).returncode:
        raise ValueError('Source has tracked changes; commit and test before preparing a pilot')
    commit = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    archive = subprocess.check_output(['git', '-C', str(repository), 'archive', '--format=tar', commit,
                                      'src', 'scripts', 'tests', 'docs', 'my_job.pmedian_nearest.alliance.sbatch'])
    destination = root/'source'
    destination.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination, filter='data')
    (root/'source_commit.txt').write_text(commit+'\n')
    return commit


def write_hashes(root, data, instances):
    root, data = Path(root), Path(data)
    paths = [p for p in (root/'source').rglob('*') if p.is_file()]
    paths += [p for p in (root/'configs').glob('*.py')]
    paths += [root/name for name in ('config.py', 'task_list.txt', 'pilot_manifest.tsv', 'source_commit.txt', 'source_data_root.txt')]
    hashes = {'files': {str(p.relative_to(root)): sha256(p) for p in sorted(paths)},
              'data': {name: sha256(data/name) for name in sorted(set(instances))}}
    (root/'pilot_sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')


def verify(root):
    root = Path(root)
    hashes = json.loads((root/'pilot_sha256.json').read_text())
    data = Path((root/'source_data_root.txt').read_text().strip())
    for base, entries in ((root, hashes['files']), (data, hashes['data'])):
        for name, expected in entries.items():
            if sha256(base/name) != expected:
                raise ValueError(f'Pilot snapshot changed: {name}')
    return (root/'source_commit.txt').read_text().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pilot_root', type=Path)
    args = parser.parse_args()
    print('Verified frozen pilot source/config/data:', verify(args.pilot_root))
