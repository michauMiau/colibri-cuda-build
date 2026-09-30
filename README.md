# Colibri with CUDA

Prebuilt Colibri binaries with CUDA support. Colibri's own repo
([JustVugg/colibri](https://github.com/JustVugg/colibri)) does not publish Linux
CUDA builds, so this repo builds one for you from each upstream release.

This is a fork of [ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda).
That project's README describes the original llama.cpp build. This workflow
builds Colibri instead, and this file is the accurate one.

## What gets built

The `qwen38` binary (Qwen3.8-Flash-Next) and the `coli` tool, from the
upstream `c/` directory. Colibri links these as standalone executables. There is
no shared library to install, so nothing goes on `LD_LIBRARY_PATH`.

Tarball layout:

```
cuda-12.8/
  qwen38              # main CLI, links libcudart
  coli                # helper tool
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

## Usage

Grab the tarball from the releases page:

```bash
tar -xzf colibri-v1.12.1-cuda-12.8-amd64.tar.gz
cd cuda-12.8

./qwen38 --help
./coli info
```

`libcudart.so.12` comes from the CUDA 12.8 runtime and has to be on the loader
path. Install the CUDA runtime system-wide, or point the linker at the one in
your toolkit.

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

## Build it yourself

```bash
git clone https://github.com/JustVugg/colibri.git
cd colibri/c
CUDA=1 CUDA_ARCH=portable make qwen38
```

## License

Colibri's license, plus the upstream ai-dock/llama.cpp-cuda license. See
[LICENSE](LICENSE).
