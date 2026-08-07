"""Bundle the package for the browser demo.

The demo ships `volsurf` as a zip that Pyodide unpacks into its filesystem.
That bundle is a *copy*, which means it can silently fall behind the package
it was made from: change a fitter, forget to rebuild, and the page keeps running
last month's code while every claim on it cites this month's numbers.

    python docs/demo/make_pkg.py        # rewrites docs/demo/volsurf-pkg.zip

`tests/test_demo_bundle.py` fails if the two ever disagree, so the drift is
caught rather than discovered.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "volsurf"
BUNDLE = ROOT / "docs" / "demo" / "volsurf-pkg.zip"
STAMP = ROOT / "docs" / "demo" / "bundle.json"


def sources(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Every .py in the package, in a stable order."""
    base = (root / "volsurf") if root else PKG
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def build(out: pathlib.Path = BUNDLE, root: pathlib.Path | None = None) -> int:
    """Zip the package for Pyodide.

    Pass `root` to bundle a checkout other than the working tree. The demo
    ships whatever is committed, so bundling uncommitted edits would put code
    on the page that is not in the repository.
    """
    files = sources(root)
    base = root or ROOT
    # Deterministic: fixed timestamps and no compression jitter, so rebuilding
    # an unchanged package produces an identical file and git stays quiet.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            info = zipfile.ZipInfo(str(path.relative_to(base)),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    # A content hash the page can hang off the asset URL. `cache: "no-cache"`
    # is a request to revalidate, and a CDN or a browser is free to answer it
    # from a stale copy; a URL that changes when the bytes change is not.
    sha = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    # vs.py is fetched separately and is not in the zip, so the zip's hash
    # cannot version it. Stamp it too, or editing the driver leaves a cached
    # copy in front of it.
    driver = out.parent / "vs.py"
    dsha = hashlib.sha256(driver.read_bytes()).hexdigest()[:8] if driver.exists() else ""
    STAMP.write_text(json.dumps(
        {"sha": sha, "driver": dsha, "modules": len(files)}) + "\n")

    # demo.js is loaded by a <script src> before any of our code runs, so it
    # cannot version itself from bundle.json. Stamp the tag at build time
    # instead, or a cached script sits in front of fresh HTML and the page
    # fails on a function that plainly exists in the file being served.
    page = out.parent / "index.html"
    js = out.parent / "demo.js"
    if page.exists() and js.exists():
        jsha = hashlib.sha256(js.read_bytes()).hexdigest()[:8]
        html = page.read_text()
        new = re.sub(r'<script src="demo\.js(\?v=[0-9a-f]+)?"></script>',
                     f'<script src="demo.js?v={jsha}"></script>', html)
        if new != html:
            page.write_text(new)
    return len(files)


if __name__ == "__main__":
    n = build()
    sha = json.loads(STAMP.read_text())["sha"]
    print(f"wrote {BUNDLE.relative_to(ROOT)} ({n} modules, "
          f"{BUNDLE.stat().st_size:,} bytes, sha {sha})")
