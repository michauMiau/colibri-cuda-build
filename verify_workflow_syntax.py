#!/usr/bin/env python3
"""Audit every step of a GitHub Actions workflow the way the runner does.

Two independent problems, checked together because both live in the same text.

1. SYNTAX. `bash -n` on the YAML `run:` value tests the wrong string: GitHub
   writes a temp script after interpolating ${{ }}, and for a `bash -c "..."`
   step the shell sees the contents of the quotes.

2. EXPANSIONS inside a `bash -c` body. The runner's shell expands `$VAR`,
   `$( )` and `${ }` in that argument BEFORE the container starts, resolving
   them against the runner's own filesystem and environment. A syntax-clean
   body can still be wrong in a way `bash -n` cannot see: `cd "$OUT"` arriving
   as `cd ""` is valid bash on every line. So a step can pass part 1 and still
   ship a red build.

   Comments are checked too, and this is not pedantry: an expansion inside a
   `#` line in that argument is expanded exactly like code. A comment quoting a
   `${VAR:+...}` append guard makes the runner emit an empty assignment and the
   container dies with `bad substitution` before running anything. Eight such
   expansions once sat in comments across three commits while a checker that
   skipped comments reported the file clean.

Usage:  verify_workflow_syntax.py [path/to/workflow.yml]
Exit 0 when every step parses, no container body holds a shell expansion, and
required job permissions are present.

ALLOWED holds lines exempted from the expansion check. Each entry needs a
mechanism in its comment, and the bad-input test must place a defect INSIDE the
exempted class -- exempting comments while never testing a comment is how the
above got through.
"""
import re
import subprocess
import sys
import tempfile

try:
    import yaml
except ImportError:
    sys.exit("needs PyYAML: pip install pyyaml")

# Jobs that push tags or releases need this, or the step 403s.
NEEDS_WRITE = {"release", "publish"}

GH_EXPR = re.compile(r"\$\{\{[^}]*\}\}")
EXPANSION = re.compile(r"\$\(|\$\{|\$[A-Za-z_][A-Za-z0-9_]*")
# $1..$9 are the runner-side docker command's own positional parameters.
POSITIONAL = re.compile(r"\$\{?\d+\}?$")

# Lines legitimately holding an expansion, with the reason.
#   export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"
# is allowed: the Makefile links with "-lcuda" and no -L, so the stubs
# directory has to reach the linker that way, the runner has no CUDA of its own
# to contribute, and the stubs path is prepended so it wins regardless.
ALLOWED = {
    'export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"',
}


def step_script(run: str) -> str:
    """What the runner's shell actually receives for this step body."""
    if 'bash -c "' in run:
        # GitHub interpolates ${{ }} but does NOT escape $(...) inside the
        # quotes, so the outer shell eats substitutions. Extract only the
        # quoted body: that is the string bash must parse.
        start = run.index('bash -c "') + len('bash -c "')
        end = run.rindex('"')
        return re.sub(r"\$\{\{[^}]*\}\}", "X", run[start:end]).replace('\\"', '"')
    return re.sub(r"\$\{\{[^}]*\}\}", "X", run)


def container_body(run: str):
    """The text between `bash -c "` and the line closing the quote, or None."""
    if 'bash -c "' not in run:
        return None
    lines = run.split("\n")
    start = next(i for i, ln in enumerate(lines) if 'bash -c "' in ln)
    closers = [i for i, ln in enumerate(lines) if ln.strip() == '"']
    if not closers:
        return None
    return "\n".join(lines[start + 1:max(closers)])


def expansions(body: str):
    """Every shell expansion in a container body, comments included."""
    hits = []
    for n, line in enumerate(body.split("\n"), 1):
        stripped = line.strip()
        if stripped in ALLOWED:
            continue
        probe = GH_EXPR.sub("GHEXPR", line)
        for m in EXPANSION.finditer(probe):
            tok = m.group(0)
            if POSITIONAL.fullmatch(tok) or tok == "$$":
                continue
            kind = "comment" if stripped.startswith("#") else "code"
            hits.append((n, tok, kind, stripped[:80]))
    return hits


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else ".github/workflows/build-cuda.yml"
    with open(path) as fh:
        wf = yaml.safe_load(fh)

    failures = 0
    checked = 0
    containers = 0

    for job_name, job in wf.get("jobs", {}).items():
        for step in job.get("steps", []):
            if "run" not in step:
                continue
            checked += 1
            label = f"{job_name}/{step.get('name', '(unnamed)')}"
            run = step["run"]
            quoted = 'bash -c "' in run

            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
                tmp.write(step_script(run))
                script_path = tmp.name
            res = subprocess.run(["bash", "-n", script_path],
                                 capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  OK    {label:58s} [{'bash -c' if quoted else 'plain'}]")
            else:
                failures += 1
                print(f"  FAIL  {label:58s} [{'bash -c' if quoted else 'plain'}]")
                print("       " + res.stderr.strip().replace("\n", "\n       ")[:300])

            body = container_body(run)
            if body is None:
                continue
            containers += 1
            hits = expansions(body)
            if not hits:
                print(f"  OK    {label:58s} [{len(body.splitlines())} lines, no expansions]")
                continue
            failures += len(hits)
            print(f"  FAIL  {label:58s} [{len(hits)} expansion(s) the runner eats]")
            for n, tok, kind, text in hits:
                print(f"         {kind:7s} line {n:3d}  {tok:12s}  {text}")

        if job_name in NEEDS_WRITE:
            perms = job.get("permissions", {})
            if perms.get("contents") != "write":
                print(f"  FAIL  {job_name}/permissions lacks contents: write")
                failures += 1
            else:
                print(f"  OK    {job_name}/permissions contents: write")

    print(f"\n{checked} steps checked, {containers} container body/bodies, "
          f"{failures} problems")
    if failures:
        print("The runner expands each reported item before the container starts.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
