# Colibri with CUDA

Prebuilt Colibri binaries with CUDA support. Colibri's own repo
([JustVugg/colibri](https://github.com/JustVugg/colibri)) does not publish Linux
CUDA builds, so this repo builds one for you from each upstream release.

This is a fork of [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda).
That project's README describes the original llama.cpp build. This workflow
builds Colibri instead, and this file is the accurate one.

## What gets built

The `qwen38` binary (Qwen3.8-Flash-Next) and the `coli` tool, from the
upstream `c/` directory. Colibri links the CLI statically, but it still needs
`libcudart.so.12` at load time, so that one shared library ships in the
tarball. Nothing else is needed from CUDA.

Tarball layout:

```
cuda-12.8/
  qwen38              # main CLI
  qwen38.run          # wrapper, only needed outside this layout
  coli                # helper tool, a Python launcher
  coli.run
  lib/
    libcudart.so.12   # CUDA 12.8 runtime, taken from the build container
  backend_cuda.o      # CUDA kernels, relocatable object
  family_registry.py  # model-family metadata
  ...                 # the upstream Python control plane and tools
```

## GPU support

Built with `CUDA_ARCH=portable`, which emits SASS for:

| Compute capability | Cards |
|---|---|
| 8.0 | A100 |
| 8.6 | RTX 3000 series |
| 8.9 | RTX 4000 series, L4, L40 |
| 9.0 | H100, H200, GH200 |
| 12.0 | RTX 5000 series, RTX Pro |

Plus a `compute_120` PTX payload, so cards newer than the table still JIT to
native code on first run.

Two older generations are deliberately missing. sm_75 (Turing) is not in the
portable list because Colibri's `c/Makefile` starts at sm_80. sm_100 (B200) is
missing because the list does not include it either, even though 12.8 can
target it. If you need either, edit `CUDA_GENCODE` in `c/Makefile` and build it
yourself, or pass your own `CUDA_ARCH`.

A toolkit of 12.9 or newer adds sm_121 (GB10 Spark), because sm_121 cannot be
compiled with 12.8.

## Requirements

- NVIDIA GPU, compute capability 8.0 or newer
- NVIDIA driver 570.15 or newer, for CUDA 12.8
- Linux x86-64
- glibc 2.34 or newer
- No CUDA toolkit. The one shared library the binary needs is in the tarball.

The binary also links `libgomp.so.1` and `libstdc++.so.6`, and neither is
bundled: `libgomp` is the OpenMP runtime and `libstdc++` belongs to the system
toolchain, so both are the system's business rather than CUDA's. A slim Debian
or Ubuntu image may not have them. If `qwen38` fails with
`error while loading shared libraries: libgomp.so.1` or the same for
`libstdc++.so.6`:

```bash
apt install libgomp1 libstdc++6   # Debian, Ubuntu
dnf install libgomp libstdc++     # Fedora, RHEL
```

`qwen38` is a glibc binary. It will not run on musl-based systems, so Alpine
needs `gcompat` and may still fail. Use a glibc distribution.

## Usage

Grab the tarball from the releases page:

```bash
tar -xzf colibri-v1.12.1-cuda-12.8-amd64.tar.gz
cd cuda-12.8

./qwen38 --help
./coli info
```

That is the whole setup. `./qwen38` finds `lib/libcudart.so.12` on its own,
because the build rewrites its RUNPATH to `$ORIGIN/lib`, and `$ORIGIN` is the
directory holding the binary. No wrapper, no `LD_LIBRARY_PATH`, no `CUDA_HOME`,
no flags.

The `.run` files are only for the case where you move a binary out of this
layout. `qwen38.run` sets `LD_LIBRARY_PATH` to the `lib/` next to it and also
exports `CUDA_HOME`, which is what Colibri's own `CUDA_HOME` lookup reads. If you
have CUDA installed system-wide and want that instead, `qwen38.run` respects an
existing `LD_LIBRARY_PATH` and appends the bundle to it.

If you would rather use your own runtime, delete `lib/` and let the system one
serve. If you get an error like

```
error while loading shared libraries: libcudart.so.12: cannot open shared object file
```

then the binary is no longer next to its `lib/`, and the `.run` wrapper is what
you want.

## Host architectures

Only `-amd64` (x86-64) is published. The arm64 matrix entry is commented out, so
there is no arm64 tarball to download.

## How the build runs

Daily at 00:00 UTC, and on demand from the Actions tab. The workflow polls
`JustVugg/colibri` for a new release tag, skips the build if that tag was
already built, and otherwise builds it, uploads the tarball, and publishes a
GitHub release under the upstream tag name.

Force a rebuild of an already-built tag with the `force_build` input.

## Verify what you downloaded

The tarball is small because Colibri links statically, not because something is
missing. To confirm the CUDA kernels are really in there:

```bash
# the CUDA fatbin section must exist
readelf -S backend_cuda.o | grep nv_fatbin

# list the architectures it was compiled for
strings backend_cuda.o | grep -oE 'sm_[0-9]{2}' | sort -u
```

If that second command prints only `sm_52`, you have a build from before the
portable-arch fix, and it will fault on the first kernel dispatch. See
[PR #1](https://github.com/michauMiau/colibri-cuda-build/pull/1).

To confirm the bundled runtime is wired up rather than just present:

```bash
# must print $ORIGIN/lib
readelf -d qwen38 | grep RUNPATH

# must not error
./qwen38 --help

# and it must stop working if you take the library away, otherwise the
# binary was reading some other copy and the bundle is not what runs it
mv lib/libcudart.so.12 /tmp/ && ./qwen38 --help; mv /tmp/libcudart.so.12 lib/
```

That last command is expected to fail with `cannot open shared object file`.
If it succeeds, the binary is picking up a system CUDA you did not intend to
depend on.

## Build it yourself

```bash
git clone https://github.com/JustVugg/colibri.git
cd colibri/c
CUDA=1 CUDA_ARCH=portable make qwen38
```

## License

Colibri's license, plus the upstream ai-dock/llama.cpp-cuda license. See
[LICENSE](LICENSE).
