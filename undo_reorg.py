"""Undo the 2026-08-08 reorganisation. Run:  python undo_reorg.py --apply
Note: the path-bootstrap headers inserted into scripts are NOT removed by this
script; they are harmless once files are back at the root (the marker file is
still found), but you may strip them manually if you prefer.
"""
import csv, shutil, sys
from pathlib import Path
rows = list(csv.DictReader(open(Path(__file__).parent / "REORG_MANIFEST.csv",
                                encoding="utf-8")))
apply = "--apply" in sys.argv
for r in reversed(rows):
    src, dst = Path(r["to"]), Path(r["from"])
    if src.exists() and not dst.exists():
        print(("restore " if apply else "would restore "), src.name)
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
print("done" if apply else "dry run — re-run with --apply")
