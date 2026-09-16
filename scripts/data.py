"""List, download, restore, and verify versioned SpiderNet data bundles."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8 * 1024 * 1024

def native(path):
    value = str(Path(path).absolute())
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return Path(value)

def sha256(path):
    with native(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def safe_target(root, name):
    rel = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe manifest path: {name!r}')
    if '.git' in rel.parts:
        raise ValueError('Data archives cannot modify Git metadata')
    root = native(root).resolve()
    path = (root / Path(*rel.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Target escapes destination: {name}')
    return path

def selected(catalog, bundles, profile):
    ids = list(bundles or [])
    if profile:
        ids += catalog['profiles'][profile]['bundles']
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError('Choose --bundle ID (repeatable) or --profile NAME')
    return [catalog['bundles'][key] for key in ids]

def restore_bundle(bundle, archive, root, overwrite=False):
    archive = native(archive)
    if not archive.is_file():
        raise FileNotFoundError(f'Archive missing: {archive.name}. Download it first or use --archive-dir.')
    if archive.stat().st_size != bundle['archive_bytes'] or sha256(archive) != bundle['archive_sha256']:
        raise ValueError(f'Archive checksum mismatch: {archive.name}')
    count = 0
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members are not allowed')
        expected = {item['archive_path'] for item in bundle['files']}
        if set(names) != expected:
            raise ValueError('ZIP inventory does not match the published manifest')
        for item in bundle['files']:
            info = z.getinfo(item['archive_path'])
            safe_target(root, info.filename)
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir() or info.file_size != item['bytes']:
                raise ValueError(f'Invalid archive member: {info.filename}')
            target = safe_target(root, item['target'])
            if target.exists():
                if target.is_file() and sha256(target) == item['sha256']:
                    continue
                if not overwrite:
                    raise FileExistsError(f'Existing file differs; preserved: {target}. Use --overwrite only deliberately.')
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.download-part')
            if temp.exists():
                raise FileExistsError(f'Partial file already exists: {temp}')
            digest = hashlib.sha256()
            try:
                with z.open(info) as source, temp.open('xb') as dest:
                    while chunk := source.read(CHUNK):
                        dest.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError(f'File checksum mismatch: {item["target"]}')
                os.replace(temp, target)
            except BaseException:
                if temp.exists(): temp.unlink()
                raise
            count += 1
    print(f'{bundle["id"]}: restored {count}; identical existing files preserved')

def download(bundle, directory):
    url = bundle.get('url')
    if not url:
        raise ValueError(f'{bundle["id"]}: not published yet; no Zenodo URL is configured. Local ZIP restoration is available.')
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ValueError('Only HTTPS download URLs are accepted')
    target = native(directory) / bundle['archive_name']
    if target.exists():
        if sha256(target) == bundle['archive_sha256']: return target
        raise ValueError(f'Existing archive has a different checksum: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.download-part')
    with urllib.request.urlopen(url, timeout=120) as response, temp.open('xb') as output:
        shutil.copyfileobj(response, output, CHUNK)
    if temp.stat().st_size != bundle['archive_bytes'] or sha256(temp) != bundle['archive_sha256']:
        raise ValueError(f'Download checksum mismatch: {temp}')
    os.replace(temp, target)
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'fetch', 'restore', 'verify'])
    parser.add_argument('--bundle', action='append', default=[])
    parser.add_argument('--profile')
    parser.add_argument('--archive-dir', type=Path, default=ROOT/'.spidernet/downloads')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace differing destination data')
    args = parser.parse_args()
    catalog = json.loads((ROOT/'data/manifest.json').read_text(encoding='utf8'))
    if args.command == 'list':
        for key, value in catalog['profiles'].items(): print(f'Profile {key}: '+', '.join(value['bundles']))
        for key, bundle in catalog['bundles'].items():
            print(f'{key}: {bundle["archive_bytes"]/1024**3:.3f} GiB, {len(bundle["files"])} files, '+('public download configured' if bundle.get('url') else 'public download pending; see docs/DATA.md for access'))
        return 0
    failures = []
    for bundle in selected(catalog, args.bundle, args.profile):
        if args.command == 'fetch':
            download(bundle, args.archive_dir)
        elif args.command == 'restore':
            restore_bundle(bundle, args.archive_dir/bundle['archive_name'], args.destination, args.overwrite)
        else:
            for item in bundle['files']:
                p = safe_target(args.destination, item['target'])
                if not p.is_file() or p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
                    failures.append(item['target'])
            print(f'Checked {bundle["id"]}')
    for path in failures: print(f'MISSING OR DIFFERENT: {path}')
    if not failures and args.command == 'restore':
        from release_environment import configure_manifests
        configure_manifests(args.destination)
    return 1 if failures else 0

if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
