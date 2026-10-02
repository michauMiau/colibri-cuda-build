#!/usr/bin/env python3
"""Execute a `docker run bash -c "..."` body locally, then assert on the result.

Why this exists. Reading a step body to decide whether it is correct is how
ordering bugs survive review: every line is individually fine and the defect
lives in the sequence. A redirection above its `mkdir`, an `ls` on a path the
same block has not created yet, a marker written beside a worktree the script
deletes at the end — each ships a red run and each is invisible in a diff.

So run it. Extract the body, execute it under `set -e` against a real
filesystem with the slow commands stubbed, and require both that it completes
and that the files a later step reads exist afterwards.

Usage:
    python3 execute_step_body.py <workflow.yml> [step-name-prefix] [required-relpath ...]

    python3 execute_step_body.py wf.yml "Build Colibri" cuda-runtime/libcudart.so.12

The required paths are relative to the fake workspace root. Defaults to the
three that the colibri-cuda-build workflow hands between steps; change them for
yours.

Two traps this script exists to have already solved:

  - Print stdout AND stderr. Bodies that use `2>&1` put diagnostics on stdout
    and leave stderr empty, so a non-zero exit with an empty stderr is
    unreadable. The first version of this harness printed only stderr and
    reported `exit 2` for many minutes against a workflow that was fine.

  - Fake the toolchain tree. A body with `ls -l /some/glob 2>&1` returns 2 when
    the glob matches nothing, and under `set -e` that ends the harness on its
    own stub. Without a CUDA tree here you will chase a bug that exists only in
    the harness.

A green result is not sufficient evidence. Feed this script a deliberately
broken body -- move a `mkdir -p` below the write that needs it -- and require it
to report the failure. A checker that has only ever passed is not a checker.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# Commands that are slow, privileged or network-bound. Every filesystem touch
# in the body is left intact, because the point is which paths get created and
# in what order.
STUBS = r"""
apt-get() { :; }
dpkg() { :; }
docker() { :; }
sudo() { :; }
nvcc() { :; }
make() { mkdir -p c; touch c/qwen38; echo "make $*"; }
ldconfig() { :; }
strip() { :; }
file() { :; }
patchelf() { :; }
tee() { cat; }
git() {
  case "$1" in
    rev-parse) echo 0123456789abcdef0123456789abcdef01234567 ;;
    clone)
      mkdir -p colibri/c colibri/c/tests colibri/c/shaders
      touch colibri/c/.gitignore
      for f in a.c b.h run.cu x.mm make.ps1 build.cmd; do touch "colibri/c/$f"; done
      echo cloned ;;
    checkout) : ;;
    *) : ;;
  esac
}
"""

# Piping into a truncator. head/tail exit early, the writer takes SIGPIPE, and
# under pipefail + set -e that ends the step after the useful output already
# printed -- so it reads as a successful partial run. Grep for the shape too:
# the stubs emit no output, so the pipe never fills and a green execution here
# does not mean the real run is safe.
PIPE_TRUNCATOR = re.compile(r'^\s*(\w[\w./-]*)\s.*\|\s*(head|tail)\b')

# Expansions the runner's own shell resolves before the container starts.
# ${{ }} is Actions, not a shell. Positional params belong to the docker
# command. Everything else is a live expansion. Comments are NOT exempt: a
# comment quoting an expansion is expanded exactly like code.
GH_EXPR = re.compile(r'\$\{\{[^}]*\}\}')
SHELL_EXPANSION = re.compile(r'\$\(|\$\{|\$[A-Za-z_][A-Za-z0-9_]*')
POSITIONAL = re.compile(r'^\$\{?\d+\}?$')

# Allowlist for expansions that are known, required and harmless, each with the
# reason recorded so a change to the line fails the audit instead of passing by
# accident.
KNOWN_OK = {}


def extract_body(run, step_name):
    """Return the text the container's shell receives."""
    if 'bash -c "' not in run:
        return run, False
    start = run.index('bash -c "') + len('bash -c "')
    lines = run[start:].split('\n')
    end = max(i for i, ln in enumerate(lines) if ln.strip() == '"')
    return '\n'.join(lines[1:end]), True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    wf = Path(sys.argv[1])
    prefix = sys.argv[2] if len(sys.argv) > 2 else 'Build'
    required = sys.argv[3:] or [
        'cuda-runtime/libcudart.so.12',
        'binaries/cuda-X/qwen38',
        'binaries/cuda-X/.commit',
    ]

    doc = yaml.safe_load(wf.read_text())
    job = next((j for name, j in doc['jobs'].items()
                if any(s.get('name', '').startswith(prefix)
                       for s in j.get('steps', []))), None)
    if job is None:
        print(f"no job has a step starting with {prefix!r}")
        return 2
    step = next(s for s in job['steps'] if s.get('name', '').startswith(prefix))
    body, quoted = extract_body(step['run'], step.get('name', ''))
    body = GH_EXPR.sub('X', body)
    print(f"step:   {step.get('name')}")
    print(f"quoted: {quoted}")
    print(f"body:   {len(body.splitlines())} lines")

    fail = 0

    # --- static checks on the raw body -------------------------------
    expansions = []
    for i, ln in enumerate(body.split('\n'), 1):
        s = ln.strip()
        probe = GH_EXPR.sub('GHEXPR', ln)
        for m in SHELL_EXPANSION.finditer(probe):
            tok = m.group(0)
            if POSITIONAL.match(tok) or tok == '$$' or s in KNOWN_OK:
                continue
            expansions.append((i, tok, ('# ' if s.startswith('#') else '') + s[:80]))

    if expansions:
        fail += 1
        print(f"  FAIL  {len(expansions)} shell expansion(s) the runner resolves "
              f"before the container starts:")
        for i, tok, ln in expansions:
            print(f"          line {i}: {tok:10s} {ln}")
    else:
        print("  OK    no shell expansions in the body (comments included)")

    sigpipes = []
    for i, ln in enumerate(body.split('\n'), 1):
        s = ln.strip()
        if not s or s.startswith('#'):
            continue
        if PIPE_TRUNCATOR.match(s) and 'tee' not in s:
            sigpipes.append((i, s[:80]))
    if sigpipes:
        fail += 1
        print("  FAIL  piped into head/tail: SIGPIPE 141 under pipefail + set -e")
        for i, ln in sigpipes:
            print(f"          line {i}: {ln}")
    else:
        print("  OK    nothing piped into head/tail")

    # --- execute ------------------------------------------------------
    work = Path(tempfile.mkdtemp(prefix='stepbody-'))
    ws = work / 'workspace'
    ws.mkdir()
    tool = work / 'toolchain'
    (tool / 'lib64' / 'stubs').mkdir(parents=True)
    # A plausible versioned library, so globs against it match and a body that
    # lists the tree has something to list.
    (tool / 'lib64' / 'libcudart.so.12.8.87').write_bytes(b'\x7fELF' + b'\0' * 4092)
    (tool / 'bin').mkdir()

    patched = (body
               .replace('/usr/local/cuda', str(tool))
               .replace('/workspace', str(ws)))
    script = work / 'body.sh'
    script.write_text(f'set -e\n{STUBS}\n{patched}\n')

    r = subprocess.run(['bash', str(script)], capture_output=True, text=True,
                       cwd=str(work))
    print(f"\nexit code: {r.returncode}")
    for label, stream in (('stdout', r.stdout), ('stderr', r.stderr)):
        if stream.strip():
            print(f"--- {label} (tail) ---")
            for ln in stream.strip().split('\n')[-15:]:
                print(f"  {ln}")

    if r.returncode != 0:
        fail += 1
        print(f"  FAIL  the body exited {r.returncode}; a real run would stop there")
    else:
        print("  OK    the body ran to completion under set -e")

    print("\n--- workspace afterwards ---")
    for p in sorted(ws.rglob('*')):
        kind = 'd' if p.is_dir() else 'f'
        size = '' if p.is_dir() else f" {p.stat().st_size}B"
        print(f"  {kind} {p.relative_to(ws)}{size}")

    print()
    for rel in required:
        if (ws / rel).exists():
            print(f"  OK      {rel}")
        else:
            fail += 1
            print(f"  MISSING {rel}   (a later step reads this)")

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nVERDICT: {fail} problem(s)")
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
