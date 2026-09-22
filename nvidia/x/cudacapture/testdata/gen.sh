#!/bin/sh
# Regenerates the PTX, CUBIN and SASS fixtures in this directory from the .cu
# sources. Run on a host with CUDA installed; set CUDA to its prefix. The build
# happens in a fixed directory so that the absolute path the disassembler
# embeds in its "//## File" comments is reproducible across hosts:
#
#	CUDA=/usr/local/cuda-13.4 ./gen.sh
#
# The generated files are byte-exact toolchain output. Do not edit them by
# hand: a fixture edited to agree with the loader proves nothing about what the
# toolchain emits. In particular the kernel names here are the real symbols --
# scale is a C++ symbol and reaches the CUBIN Itanium-mangled, fill is
# extern "C" and does not -- and LoadCode joins the three tiers on them.
#
# Two units are built:
#
#	kernels.*	one file, two entry points, both linkages
#	main.*		two files, with the entry point in the included one
#	inline.*	one file plus a header, force-inlined into the kernel
#
# The CUBINs are stored hex-encoded, as every other CUBIN fixture in this
# repository is, so that the tree holds no binary files; the tests decode them
# on read.
set -e
: "${CUDA:=/usr/local/cuda}"
BIN="$CUDA/bin"
BUILD=/tmp/cudacapture-fixtures
SRC=$(cd "$(dirname "$0")" && pwd)

rm -rf "$BUILD"
mkdir -p "$BUILD"
cp "$SRC"/*.cu "$SRC"/*.cuh "$BUILD"
cd "$BUILD"

for unit in kernels main inline; do
	"$BIN/nvcc" -lineinfo -ptx -arch=sm_90 "$unit.cu" -o "$SRC/$unit.ptx"
	"$BIN/ptxas" -arch=sm_90 -lineinfo "$SRC/$unit.ptx" -o "$unit.cubin"
	"$BIN/nvdisasm" -c -g "$unit.cubin" > "$SRC/$unit.sass"
	xxd -p -c 1000000 "$unit.cubin" > "$SRC/$unit.cubin.hex"
done

"$BIN/nvcc" --version | tail -2
