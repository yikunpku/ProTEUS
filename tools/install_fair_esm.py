#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Install Meta's ESM (`fair-esm`) under the import name `fair_esm`.

Two different projects both ship a top-level package called `esm`:

    fair-esm 2.0.0          Meta        ESM2, ESM-MSA-1b, ESM-1v, ESM-IF1
    esm      3.x            EvolutionaryScale   ESM3, ESM C

Proteus needs both. `pip install fair-esm esm` cannot give you both: whichever
is installed second overwrites the other's `esm/` directory, and pip will not
warn you. So Proteus imports Meta's as `fair_esm` and EvolutionaryScale's as
`esm`, and this script is what makes `fair_esm` exist.

It installs fair-esm into a scratch directory (never into site-packages, so an
existing `esm/` is untouched), renames the package, rewrites the module
references in its own source, and drops the result next to your other
packages. Run it after `pip install -e .`:

    python tools/install_fair_esm.py

It is idempotent: if `import fair_esm` already works it does nothing. Use
--force to reinstall anyway.
"""

import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

VERSION = "fair-esm==2.0.0"

# fair-esm refers to itself by absolute name, in imports (`from esm.data
# import ...`) and in annotations and call sites alike (`esm.data.Alphabet`),
# so every module reference has to move -- renaming only the imports leaves
# `NameError: name 'esm' is not defined` at class-body evaluation time.
#
# Both patterns refuse a leading `_`/word character, so `fair_esm` is never
# doubled and a word that merely starts with "esm" (`esmfold`, `esm2_t36`) is
# left alone; refusing a leading quote protects the one string literal that
# contains "esm." -- an ESMFold state-dict key prefix, not a module path.
_LEAD = r"(?<![_\w\"'])"
MODULE_REF = re.compile(_LEAD + r"esm(?=\.)")
BARE_IMPORT = re.compile(_LEAD + r"(from|import)(\s+)esm(?![\w.])")


def download_command(staging: Path) -> list:
    """Build the command that fetches fair-esm into `staging`.

    `python -m pip` is not always there: a `uv venv` has no pip at all, and uv
    is now a common way to install this project. So prefer pip when the
    interpreter has it, fall back to `uv pip --python <this interpreter>`, and
    say what to do when neither exists rather than failing on a bare
    `No module named pip`.
    """
    base = ["--quiet", "--no-deps", "--target", str(staging), VERSION]
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install"] + base
    uv = shutil.which("uv")
    if uv:
        # `--python` belongs to the subcommand, not to `uv pip`.
        return [uv, "pip", "install", "--python", sys.executable] + base
    raise SystemExit(
        "neither pip nor uv is available to this interpreter "
        f"({sys.executable}), so fair-esm cannot be fetched. Install one of "
        "them into the environment (`uv pip install pip`, or put `uv` on "
        "PATH) and run this again.")


def site_packages() -> Path:
    """Where to put the package: the same directory pip just installed into."""
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path and Path(path).is_dir():
            return Path(path)
    raise SystemExit("cannot locate site-packages for this interpreter")


def rewrite(root: Path) -> int:
    """Rename the package's self-references, without touching shared copies.

    Every write goes to a temporary file that then replaces the original, so
    the original inode is left alone. Writing in place would be wrong: both uv
    and pip populate a target directory with hardlinks into their own unpacked
    wheel cache, so truncating the staged file edits the cache entry too. The
    effect was real and silent -- after one run, this machine's uv cache held a
    fair-esm whose modules had been renamed, so any later `uv pip install
    fair-esm`, in any project, would hand out a package called `esm` with
    `fair_esm` imports inside it.
    """
    changed = 0
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        new = BARE_IMPORT.sub(r"\1\2fair_esm", MODULE_REF.sub("fair_esm", text))
        if new == text:
            continue
        tmp = py.with_name(py.name + ".proteus-tmp")
        tmp.write_text(new, encoding="utf-8")
        tmp.replace(py)
        changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="reinstall even if fair_esm already imports")
    args = ap.parse_args()

    target = site_packages() / "fair_esm"
    if not args.force and importlib.util.find_spec("fair_esm") is not None:
        print("fair_esm already importable; nothing to do (--force to redo)")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        cmd = download_command(staging)
        print(f"installing {VERSION} into a scratch directory "
              f"(via {Path(cmd[0]).name})")
        subprocess.check_call(cmd)

        src = staging / "esm"
        if not src.is_dir():
            raise SystemExit(f"{VERSION} did not provide an 'esm' package in {staging}")

        n = rewrite(src)
        print(f"rewrote self-imports in {n} files")

        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src), str(target))

        # Keep the dist metadata so `pip list` and proteus-doctor can see it.
        for meta in staging.glob("fair_esm-*.dist-info"):
            dest = site_packages() / meta.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(meta), str(dest))

    print(f"installed -> {target}")
    out = subprocess.run(
        [sys.executable, "-c",
         "import fair_esm, fair_esm.pretrained; print('fair_esm', fair_esm.__version__)"],
        capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
