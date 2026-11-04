# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Check that this machine can run Proteus, and say what to do if it cannot.

Each check reports what it found, not just pass/fail, because the useful
information is usually the version or the path. Behaviour is probed directly
where a version number would not tell the truth -- notably the MSA batch
converter, which some installs patch.
"""

import argparse
import importlib
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REQUIRED = {
    "torch": "2.1",
    "numpy": "1.26",
    "pandas": "1.5",
    "scipy": "1.10",
    "sklearn": None,
    "esm": "3.0",
    "Bio": None,
    "numba": None,
    "einops": None,
    "yaml": None,
    "huggingface_hub": None,
    "tqdm": None,
}

OPTIONAL = {
    "biotite": "reads structures out of PDB files, for structure tokenization "
               "and the structure-conditioned zero-shot channel. "
               "pip install 'proteus[structure]'.",
}

FAIR_ESM_HINT = ("run `python tools/install_fair_esm.py`. It cannot come from "
                 "a plain `pip install fair-esm`: that package's top-level "
                 "name is `esm`, which collides with EvolutionaryScale's.")


class Report:
    def __init__(self):
        self.rows: List[Tuple[str, bool, str]] = []
        self.warnings: List[str] = []

    def ok(self, name: str, detail: str = ""):
        self.rows.append((name, True, detail))
        print(f"  [ ok ] {name:34} {detail}")

    def fail(self, name: str, detail: str):
        self.rows.append((name, False, detail))
        print(f"  [FAIL] {name:34} {detail}")

    def warn(self, name: str, detail: str):
        self.warnings.append(name)
        print(f"  [warn] {name:34} {detail}")

    @property
    def failures(self) -> List[str]:
        return [n for n, ok, _ in self.rows if not ok]


def _version(module) -> str:
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(module, attr, None)
        if isinstance(v, str):
            return v
    return "unknown"


def check_packages(rep: Report) -> None:
    print("\npackages")
    for name, minimum in REQUIRED.items():
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            rep.fail(name, f"not importable ({exc}); install the project deps")
            continue
        version = _version(mod)
        if minimum and version != "unknown":
            have = tuple(int(x) for x in version.split(".")[:2] if x.isdigit())
            want = tuple(int(x) for x in minimum.split("."))
            if have and have < want:
                rep.fail(name, f"{version} < {minimum}")
                continue
        rep.ok(name, version)

    for name, hint in OPTIONAL.items():
        try:
            mod = importlib.import_module(name)
            rep.ok(f"{name} (optional)", _version(mod))
        except ImportError:
            rep.warn(f"{name} (optional)", f"absent. {hint}")


def check_esm_namespaces(rep: Report) -> None:
    """Two different projects both ship a top-level package called `esm`.

    `fair-esm` is Meta's ESM-1/ESM-2/ESM-MSA-1b; `esm` 3.x is
    EvolutionaryScale's ESM3 and ESM C. Proteus needs both, and pip cannot
    install both: whichever lands second overwrites the other's `esm/`
    directory without a word. So Meta's is installed under the name `fair_esm`
    (`tools/install_fair_esm.py`) and `esm` is left to EvolutionaryScale.

    This matters beyond install time. `torch.hub.load("facebookresearch/esm",
    ...)` does `from esm.pretrained import ...`, which on a machine with both
    resolves to whichever owns the name -- so it can load the wrong library
    while looking like it worked. Proteus never uses torch.hub for this reason.

    Both halves are checked by what they actually contain, not by version
    string, so a fair-esm that has taken over the `esm` name is caught.
    """
    print("\nESM namespaces")
    try:
        import fair_esm
    except ImportError:
        rep.fail("fair_esm present", f"missing; ESM2 and ESM-MSA-1b come from it. {FAIR_ESM_HINT}")
    else:
        if importlib.util.find_spec("fair_esm.pretrained") is None:
            rep.fail("fair_esm is Meta's ESM", "no fair_esm.pretrained module")
        else:
            rep.ok("fair_esm", f"{_version(fair_esm)} (Meta)")

    try:
        import esm
    except ImportError:
        rep.fail("esm present", "missing; ESM3 is imported at module level. pip install 'esm>=3.0'")
        return
    # fair-esm installed the ordinary way lands here instead, and every ESM3
    # path then fails with a confusing ModuleNotFoundError deep in an import.
    if importlib.util.find_spec("esm.models.esm3") is None:
        rep.fail("esm is EvolutionaryScale's",
                 f"`esm` {_version(esm)} has no esm.models.esm3 -- this looks "
                 f"like fair-esm occupying the name. Reinstall: "
                 f"pip install --force-reinstall 'esm>=3.0', then {FAIR_ESM_HINT}")
    else:
        rep.ok("esm", f"{_version(esm)} (EvolutionaryScale)")


def check_msa_batch_converter(rep: Report) -> None:
    """Probe the MSA batch converter's padding width.

    Stock fair-esm pads an MSA to its own alignment width. Some installs patch
    `MSABatchConverter.__call__` to hardcode `max_seqlen = 512`; we found one,
    which is why this probe exists.

    Which path this affects, because the two MSA consumers do not share a
    copy of fair-esm. `proteus.featurize.msa` imports the installed
    `fair_esm`, so it gets this converter. The zero-shot scorer does not: it
    loads the MSA model through `proteus.third_party.esm_fork`, which carries
    its own unpatched `MSABatchConverter`, so its masked-marginal loop runs
    once per real alignment column whatever is installed here (measured: 120
    columns for a 119-residue query, prepend_bos and no append_eos).

    For `featurize.msa` the patch means the encoder runs on a tensor padded to
    513 columns rather than the alignment's own width. At 512 columns or fewer
    the positions that are read back are unchanged, so the embeddings match;
    past 512 the alignment is truncated and they do not.

    So this is reported, not corrected. Patching third-party code from here
    would make Proteus's results depend on a modification nobody declared.
    """
    print("\nMSA batch converter")
    try:
        import fair_esm
        alphabet = fair_esm.data.Alphabet.from_architecture("msa_transformer")
        rows = [("q", "MAGLRHTFVV"), ("a", "-AGLRHTFVV")]
        _, _, tokens = alphabet.get_batch_converter()([rows])
        width = int(tokens.shape[2])
        expected = len(rows[0][1]) + int(alphabet.prepend_bos) + int(alphabet.append_eos)
        if width == expected:
            rep.ok("pads to the alignment width", f"{width} columns for a {len(rows[0][1])}-column MSA")
        else:
            rep.warn("pads to a fixed width",
                     f"{width} columns for a {len(rows[0][1])}-column MSA "
                     f"(expected {expected}): this install has patched "
                     f"MSABatchConverter to a constant. It affects "
                     f"`featurize.msa`, which uses the installed fair-esm; the "
                     f"zero-shot scorer loads its own unpatched copy from "
                     f"proteus.third_party.esm_fork and is unaffected. An "
                     f"alignment of <={width - 1} columns is padded rather than "
                     f"changed, so those embeddings still match stock "
                     f"fair-esm, just computed on a wider tensor; a longer one "
                     f"is truncated and will NOT match. Reinstall with "
                     f"`python tools/install_fair_esm.py --force`.")
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        rep.fail("MSA batch converter probe", f"{type(exc).__name__}: {exc}. {FAIR_ESM_HINT}")


def check_msa_weights(rep: Report) -> None:
    """Check that the cached MSA-Transformer file is the published one.

    `torch.hub`'s checkpoint directory is a flat namespace keyed by filename,
    so anything that drops a file called `esm_msa1b_t12_100M_UR50S.pt` there --
    another project's vendored copy, a symlink -- is what `fair_esm` will load,
    silently. We hit exactly that: a file of the right name whose weights were
    unrelated to the published model (zero tensors in common, median cosine
    0.0008), differing in one shape that is cheap to detect.
    """
    print("\nMSA weights identity")
    try:
        import torch
        from pathlib import Path
        path = Path(torch.hub.get_dir()) / "checkpoints" / "esm_msa1b_t12_100M_UR50S.pt"
        if not path.exists():
            rep.warn("MSA checkpoint", "not cached yet; will download on first use")
            return
        if path.is_symlink():
            rep.warn("MSA checkpoint is a symlink",
                     f"-> {path.resolve()}. torch.hub's cache is keyed by "
                     f"filename only, so verify this really is the published "
                     f"ESM-MSA-1b.")
        state = torch.load(path, map_location="meta", mmap=True,
                           weights_only=False)["model"]
        key = "encoder.sentence_encoder.msa_position_embedding"
        if key not in state:
            rep.fail("MSA checkpoint layout", f"{key} absent; this is not an "
                                              f"MSA-Transformer checkpoint")
            return
        shape = tuple(state[key].shape)
        embed_dim = int(state["encoder.sentence_encoder.embed_tokens.weight"].shape[1])
        if shape[-1] == embed_dim:
            rep.ok("MSA checkpoint", f"{key[-24:]} {shape}, embed_dim {embed_dim}")
        else:
            rep.fail("MSA checkpoint",
                     f"{key} has shape {shape} but embed_dim is {embed_dim}. "
                     f"The published ESM-MSA-1b has {shape[:-1] + (embed_dim,)}. "
                     f"Delete {path} and let fair-esm download it again.")
    except Exception as exc:  # noqa: BLE001
        rep.fail("MSA checkpoint probe", f"{type(exc).__name__}: {exc}")


def _find_binary(name: str) -> Tuple[Optional[str], str]:
    """Look on PATH, then next to this interpreter.

    The second place matters: an environment installed from
    `environment.yml` has both binaries in its `bin/`, but someone who runs
    `/path/to/env/bin/proteus-doctor` without activating the environment has
    that `bin/` nowhere on PATH. Reporting "not installed" there would be
    wrong -- the binary is present, it just will not be found by a subprocess
    either, which is the thing worth saying.
    """
    found = shutil.which(name)
    if found:
        return found, "on PATH"
    sibling = Path(sys.executable).parent / name
    if sibling.exists():
        return str(sibling), "not on PATH"
    return None, "missing"


def _run_binary(path: str, args: List[str], marker: str) -> Tuple[bool, str]:
    """Execute the binary and look for `marker` in its output.

    Finding a file on PATH says nothing about whether it works. Debian's
    `pymol` is a shell script that runs a bare `python3 -m pymol.__init__`, so
    it picks up whichever python comes first -- and inside a virtualenv that
    is a python with no `pymol` module, which makes a present, executable,
    correctly-named binary fail every call. That is a worse state than not
    having it, because everything upstream reports it as installed.
    """
    import subprocess
    try:
        p = subprocess.run([path] + args, capture_output=True, text=True,
                           timeout=120)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except subprocess.TimeoutExpired:
        return False, "did not finish within 120 s"
    out = (p.stdout or "") + (p.stderr or "")
    if marker in out:
        return True, ""
    tail = " / ".join(line.strip() for line in out.strip().splitlines()[-3:])
    return False, f"exit {p.returncode}, no {marker!r} in output: {tail[:300]}"


def check_binaries(rep: Report) -> None:
    print("\nexternal binaries")
    probes = (
        ("pymol", ["-cq", "-d", "print('PYMOL_RUNS')"], "PYMOL_RUNS",
         "needed to build mutant structures",
         "conda install -c conda-forge pymol-open-source, or use the "
         "Docker image"),
        ("hhfilter", ["-h"], "HHfilter",
         "MSA filtering will be skipped",
         "conda install -c bioconda hhsuite, or use the Docker image"),
    )
    for name, args, marker, why, fix in probes:
        path, where = _find_binary(name)
        if path is None:
            rep.warn(name, f"not installed; {why}. {fix}")
            continue
        works, detail = _run_binary(path, args, marker)
        if not works:
            rep.fail(name, f"found at {path} but it does not run: {detail}. "
                           f"A broken binary is worse than a missing one -- "
                           f"{why}, and every caller will report it as "
                           f"installed.")
        elif where == "not on PATH":
            rep.warn(name, f"runs, at {path}, but that directory is not on "
                           f"PATH, so the pipeline's subprocess calls will "
                           f"not find it; {why}. Activate the environment, or "
                           f"add that directory to PATH.")
        else:
            rep.ok(name, f"{path} (runs)")


WEIGHT_HOSTS = ("huggingface.co", "dl.fbaipublicfiles.com")

TLS_HINT = ("your CA bundle cannot verify this host. Fixes, in order of "
            "preference: `pip install -U certifi`; point SSL_CERT_FILE and "
            "REQUESTS_CA_BUNDLE at a bundle that includes your proxy's root "
            "CA; or, on a conda env, `conda install -c conda-forge "
            "ca-certificates`. Do not disable verification globally -- an "
            "earlier version of this package did, and it silently turned off "
            "certificate checking for every stdlib HTTPS client in any "
            "process that imported it.")


def check_tls(rep: Report) -> None:
    """Can this machine verify the hosts the weights come from?

    Worth a check of its own because the failure is confusing: a download dies
    with CERTIFICATE_VERIFY_FAILED deep inside torch.hub, and the obvious
    workaround -- disabling verification -- is the wrong fix.
    """
    import os
    import socket
    import ssl

    print("\nTLS")
    ctx = ssl._create_default_https_context()
    if ctx.verify_mode == ssl.CERT_NONE or not ctx.check_hostname:
        rep.fail("stdlib HTTPS verification",
                 "something in this process has set "
                 "ssl._create_default_https_context to an unverified context. "
                 "Downloads would accept any certificate. Find it with "
                 "`python -X importtime -c 'import proteus'` and remove it.")
    else:
        rep.ok("stdlib HTTPS verification", "certificates are verified")

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        rep.ok("weight hosts", "skipped (HF_HUB_OFFLINE=1)")
        return

    # Probe with a context that definitely verifies, not with the default one:
    # if the default has been tampered with, every host would "pass".
    probe = ssl.create_default_context()
    for host in WEIGHT_HOSTS:
        try:
            with socket.create_connection((host, 443), timeout=5) as sock:
                with probe.wrap_socket(sock, server_hostname=host):
                    pass
            rep.ok(f"{host}", "certificate verified")
        except ssl.SSLCertVerificationError as exc:
            rep.fail(f"{host}", f"{exc.verify_message or exc}. {TLS_HINT}")
        except (socket.timeout, socket.gaierror, OSError) as exc:
            rep.warn(f"{host}", f"not reachable ({type(exc).__name__}); "
                                f"cannot tell whether TLS would verify. Fine "
                                f"if the weights are already cached.")


def check_prepare_backends(rep: Report) -> None:
    """Can `proteus-prepare` reach an MSA server and a structure predictor?

    Neither is needed to run the pipeline on inputs you already have, so a
    missing one is a warning. The structure predictor deliberately lives in
    another environment, which is why this looks for a launcher rather than
    trying to import it.
    """
    import socket
    import ssl
    import subprocess

    print("\nInput preparation (proteus-prepare)")

    host = "api.colabfold.com"
    probe = ssl.create_default_context()
    try:
        with socket.create_connection((host, 443), timeout=5) as sock:
            with probe.wrap_socket(sock, server_hostname=host):
                pass
        rep.ok("MSA server", f"{host} reachable")
    except Exception as exc:  # noqa: BLE001 - any failure means the same thing
        rep.warn("MSA server",
                 f"{host} not reachable ({type(exc).__name__}). "
                 f"--msa colabfold will not work; --msa af3-zip and "
                 f"--msa a3m do not need the network.")

    try:
        from proteus.prepare.structure import find_boltz

        cmd = find_boltz()
    except SystemExit:
        rep.warn("structure predictor",
                 "no `boltz` launcher found. Set PROTEUS_BOLTZ to the one in "
                 "its own environment, or import a structure with "
                 "--structure af3-zip / pdb / rcsb instead of predicting one.")
        return
    except Exception as exc:  # noqa: BLE001
        rep.warn("structure predictor", f"could not be checked: {exc}")
        return

    rep.ok("structure predictor", cmd[0])
    # Its environment is the point of the separation, so report what is in it.
    try:
        python = Path(cmd[0]).parent / "python"
        if python.exists():
            out = subprocess.run(
                [str(python), "-c",
                 "import torch, warp; print(torch.__version__, warp.__version__)"],
                capture_output=True, text=True, timeout=120)
            if out.returncode == 0:
                torch_v, warp_v = out.stdout.split()
                rep.ok("predictor environment",
                       f"torch {torch_v}, warp {warp_v}")
            elif "warp" in out.stderr:
                rep.warn("predictor environment",
                         "torch is there but `warp` is not. The Gauss-Seidel "
                         "projection (--project_pbd) needs it; the predictor "
                         "does not declare it as a dependency.")
            else:
                rep.warn("predictor environment",
                         f"could not be queried: {out.stderr.strip()[:120]}")
    except Exception as exc:  # noqa: BLE001
        rep.warn("predictor environment", f"could not be queried: {exc}")


def check_registry(rep: Report, config: str) -> None:
    """Do each configured protein's three descriptions agree with each other?

    The checks that `proteus-prepare` runs before it registers anything, re-run
    against whatever is on disk now. This is the one place that would catch a
    structure whose numbering stopped matching the library.
    """
    from proteus.prepare.validate import check_registry as run_checks
    from proteus.config import ProteusConfig

    print(f"\nConfigured proteins ({config})")
    try:
        cfg = ProteusConfig.from_yaml(config)
    except Exception as exc:  # noqa: BLE001
        rep.fail("config", f"{config} could not be loaded: {exc}")
        return

    results = run_checks(cfg, verbose=False)
    for key, issues in results.items():
        if issues.errors:
            rep.fail(f"inputs for {key}", "; ".join(issues.errors)[:200])
        elif issues.warnings:
            rep.warn(f"inputs for {key}", "; ".join(issues.warnings)[:200])
        else:
            rep.ok(f"inputs for {key}", "alignment, structure and sequence agree")


def check_torch(rep: Report) -> None:
    print("\ncompute")
    try:
        import torch
    except ImportError:
        rep.fail("torch", "not importable")
        return
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        detail = []
        for i in range(n):
            free, total = torch.cuda.mem_get_info(i)
            detail.append(f"cuda:{i} {torch.cuda.get_device_name(i)} "
                          f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free")
        rep.ok(f"{n} CUDA device(s)", "; ".join(detail))
        # ESM2-3B in fp32 is the binding constraint for zero-shot scoring.
        if all(torch.cuda.mem_get_info(i)[0] < 13 * 2**30 for i in range(n)):
            rep.warn("GPU headroom",
                     "every device has under 13 GiB free; ESM2-3B needs ~11.4 GiB "
                     "in fp32. Use --dtype float16, a smaller ESM2, or free memory.")
    else:
        rep.warn("CUDA", "no GPU visible. Everything runs on CPU, but zero-shot "
                         "scoring is impractical there: a 119-residue protein "
                         "takes hours because each residue needs its own "
                         "forward pass per MSA sample.")


def check_model_cache(rep: Report) -> None:
    print("\nmodel weights")
    try:
        import fair_esm.pretrained  # noqa: F401
        import torch
        hub = torch.hub.get_dir()
        from pathlib import Path
        ckpt_dir = Path(hub) / "checkpoints"
        wanted = {
            "esm2_t36_3B_UR50D": "sequence encoder",
            "esm_msa1b_t12_100M_UR50S": "MSA encoder",
        }
        for name, role in wanted.items():
            hits = list(ckpt_dir.glob(f"{name}*.pt")) if ckpt_dir.exists() else []
            if hits:
                size = sum(p.stat().st_size for p in hits) / 2**30
                rep.ok(f"{name}", f"cached, {size:.1f} GiB ({role})")
            else:
                rep.warn(f"{name}", f"not cached; will download on first use ({role})")
    except Exception as exc:  # noqa: BLE001
        rep.fail("model cache probe", f"{type(exc).__name__}: {exc}. {FAIR_ESM_HINT}")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="also check every protein in this Proteus config: do "
                         "its alignment, structure and library sequence agree?")
    args = ap.parse_args(argv)
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    rep = Report()
    check_packages(rep)
    check_esm_namespaces(rep)
    check_msa_batch_converter(rep)
    check_msa_weights(rep)
    check_binaries(rep)
    check_tls(rep)
    check_prepare_backends(rep)
    check_torch(rep)
    check_model_cache(rep)
    if args.config:
        check_registry(rep, args.config)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in rep.rows if ok)
    print(f"{passed}/{len(rep.rows)} checks passed, {len(rep.warnings)} warning(s)")
    if rep.failures:
        print(f"blocking problems: {rep.failures}")
        sys.exit(1)
    print("ready")


if __name__ == "__main__":
    main()
