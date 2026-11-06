#!/usr/bin/env python
"""Build the artist-level split manifest for the released corpus from the research split.

The released corpus (`inhouse_crepe_separated`, romanised file names) and the research corpus
(`drive_crepe_separated`, Korean file names) contain byte-identical CSVs, so files are matched by
content hash (md5). The research split CSV lists `path,artist,split` on research paths; files that
were excluded from training (non-singing clips) are written with split=exclude so that the loader
can skip them while still requiring every file on disk to be listed.

Usage (requires both corpora on disk; end users do not need to run this):
    python scripts/make_manifest.py \
        --release_root data/inhouse_data/inhouse_crepe_separated \
        --research_root /path/to/drive_crepe_separated \
        --exclude_root  /path/to/drive_data/exclude/crepe \
        --split_csv     /path/to/split_by_artist.csv \
        --out splits/inhouse_crepe_split_artist.csv
"""
import argparse, csv, hashlib, unicodedata
from pathlib import Path


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def artist_of(release_rel_path: str) -> str:
    # released corpus layout: danga/<artist>/<album>/<file>  |  madang*/<madang>/<artist>/<album...>/<file>
    parts = Path(release_rel_path).parts
    return parts[2] if parts[0].startswith('madang') else parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release_root', required=True); ap.add_argument('--research_root', required=True)
    ap.add_argument('--exclude_root', default=None, help='dir of research-corpus files excluded from training')
    ap.add_argument('--split_csv', required=True, help='research split: columns path,artist,split (paths relative to research_root)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rel_root, res_root = Path(a.release_root), Path(a.research_root)

    release = {}
    for p in sorted(rel_root.rglob('*.csv')):
        h = md5(p); assert h not in release, f'duplicate content in release corpus: {p}'
        release[h] = p.relative_to(rel_root).as_posix()

    with open(a.split_csv, newline='', encoding='utf-8') as f:
        split_rows = {unicodedata.normalize('NFC', r['path']): (r['artist'], r['split']) for r in csv.DictReader(f)}

    rows, used, folder_votes = [], set(), {}
    for p in sorted(res_root.rglob('*.csv')):
        rel = unicodedata.normalize('NFC', p.relative_to(res_root).as_posix())
        artist_kr, split = split_rows[rel]
        h = md5(p); assert h in release, f'research file has no release counterpart: {rel}'
        rel_path = release[h]
        rows.append([rel_path, artist_kr, split]); used.add(h)
        if artist_of(rel_path) != 'hapchang':   # 'hapchang' groups ensemble recordings; the split labels them by lead singer
            folder_votes.setdefault(artist_kr, {}).setdefault(artist_of(rel_path), 0); folder_votes[artist_kr][artist_of(rel_path)] += 1
    if a.exclude_root:
        for p in sorted(Path(a.exclude_root).rglob('*.csv')):
            h = md5(p); assert h in release, f'excluded file has no release counterpart: {p.name}'
            rel_path = release[h]; rows.append([rel_path, artist_of(rel_path), 'exclude']); used.add(h)   # no research label -> folder name
    # romanised artist label = the singer's own folder name in the released corpus (majority vote over their files);
    # singers that only appear inside 'hapchang' fall back to korean_romanizer.
    def romanise(kr: str) -> str:
        if kr in folder_votes:
            return max(folder_votes[kr].items(), key=lambda kv: kv[1])[0]
        from korean_romanizer.romanizer import Romanizer
        return Romanizer(kr).romanize().replace(' ', '').lower()
    rom_of = {kr: romanise(kr) for kr in {r[1] for r in rows if r[2] != 'exclude'}}
    rom2kr = {}
    for kr, rom in rom_of.items(): rom2kr.setdefault(rom, set()).add(kr)
    bad = {k: v for k, v in rom2kr.items() if len(v) > 1}
    assert not bad, f'romanised artist collides across singers: {bad}'
    for r in rows:
        if r[2] != 'exclude': r[1] = rom_of[r[1]]
    unmatched = [v for k, v in release.items() if k not in used]
    assert not unmatched, f'release files without a split: {unmatched[:3]}'

    rows = sorted(tuple(r) for r in rows)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['path', 'artist', 'split']); w.writerows(rows)
    from collections import Counter
    print(f'wrote {len(rows)} rows -> {a.out}', dict(Counter(r[2] for r in rows)), f'artists={len({r[1] for r in rows if r[2] != "exclude"})}')


if __name__ == '__main__':
    main()
