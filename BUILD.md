# Building niimath for lightprep

lightprep runs on [niimath](https://github.com/rordenlab/niimath). The binary is
not in this repository — it is a platform-specific build artifact, and a
committed one would be wrong for every machine but the one that built it,
failing in a way that reads as a lightprep bug rather than a missing dependency.

## Where lightprep looks

`lightprep._niimath.niimath_path()` checks two places, in order:

1. **`lightprep/niimath`**, beside the package — a build dropped here wins, which
   keeps a checkout self-contained and pinned to a known version;
2. **`niimath` on PATH**.

If neither is usable it raises `DependencyError` saying so.

## What version

A build recent enough to carry **`-moco`**, **`--medic`** and **`-unwarp`**.
These are what the default methods are built on, and none of them exist in
older releases:

| feature | used by |
|---|---|
| `-moco` | `lightprep.hmc.moco` — the default head motion correction |
| `--medic` | `lightprep.sdc.medic_niimath` — the default distortion correction |
| `-unwarp` | `lightprep.resample.apply_sdc_niimath` |
| `-allineate` | `lightprep.hmc.allineate`, `lightprep.coreg.allineate` |

The reference build is `v1.0.20260726`. Check what you have with:

```python
from lightprep._niimath import niimath_path, version
print(niimath_path(), version())
```

## The reference build

macOS, Apple Silicon, OpenMP statically linked so the result depends on nothing
outside macOS itself:

```sh
brew install libomp
git clone https://github.com/rordenlab/niimath.git
cd niimath/src
make ZSTD_FOUND=0 OMPLINK=/opt/homebrew/opt/libomp/lib/libomp.a -j8
cp niimath /path/to/lightprep/niimath
```

Verify it links only against system libraries:

```sh
otool -L niimath
#   /usr/lib/libSystem.B.dylib
#   /usr/lib/libz.1.dylib
```

`ZSTD_FOUND=0` drops the Homebrew zstd dependency; `OMPLINK=...libomp.a` links
libomp statically instead of dynamically. Both exist to keep the binary
self-contained — without them it needs Homebrew present at runtime.

### Why OpenMP is worth the trouble

`-allineate` is one registration per frame, and threading is most of what makes
it tolerable. Measured on 2.8 mm EPI, 8 cores:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| per registration | 9.1 s | 4.8 s | 2.7 s | 2.1 s |

Control it with `OMP_NUM_THREADS`. Note that niimath's own `-p` flag is *not*
accepted in the position its help implies — it is parsed as an input filename —
so the environment variable is the way.

`lightprep.hmc.moco`, the default, is fast either way: it estimates a 138-frame
run in about 20 s because it is a local optimizer, not a per-frame search.

### Portability caveat

Homebrew ships `libomp` for arm64 only, so the recipe above produces an
**arm64-only** binary. For a universal build, drop OpenMP and `lipo` the two
architectures together:

```sh
for A in arm64 x86_64; do
  make clean
  make ZSTD_FOUND=0 OMP=0 CNAME="clang -arch $A" -j8
  cp niimath ../niimath-$A
done
lipo -create ../niimath-arm64 ../niimath-x86_64 -output ../niimath
```

## Other platforms

Linux and Intel macOS build the same way; see niimath's own README. Plain
`make` in `src/` works and picks up OpenMP where the toolchain provides it.
lightprep does not care how the binary got there, only that it is at one of the
two locations above and is recent enough.
