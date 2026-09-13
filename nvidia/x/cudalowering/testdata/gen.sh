#!/bin/sh
# Regenerates the .ptx and .sass fixtures in this directory from the .cu
# sources. Run on a host with CUDA installed; set CUDA to its prefix:
#
#	CUDA=/usr/local/cuda-13.0 ./gen.sh
#
# The generated files are byte-exact toolchain output. Do not edit them by hand:
# a fixture edited to agree with the comparison proves nothing about what ptxas
# actually did. Each variant exists to expose one lowering decision:
#
#	lowering.sm90.sass		 default ptxas; the PTX and the SASS agree
#	lowering.sm90.rr24.sass	 register budget cut until ptxas spills
#	lowering.sm90.rr24.life.sass	 the same build with live register counts
#	lowering.sm90.demangled.sass	 the same SASS with C++ names decoded
#	lowering.sm121a.sass	 the same kernels for Blackwell, where ptxas
#				 pads the double-precision pipe with NOPs
#	other.sm90.sass		 a different module, matching no kernel here
set -e
: "${CUDA:=/usr/local/cuda}"
BIN="$CUDA/bin"
BUILD=/tmp/cudalowering-fixtures
SRC=$(cd "$(dirname "$0")" && pwd)

rm -rf "$BUILD"
mkdir -p "$BUILD"
cp "$SRC"/*.cu "$BUILD"
cd "$BUILD"

"$BIN/nvcc" -lineinfo -ptx -arch=sm_90 lowering.cu -o "$SRC/lowering.sm90.ptx"
"$BIN/ptxas" -arch=sm_90 -lineinfo "$SRC/lowering.sm90.ptx" -o lowering.cubin
"$BIN/nvdisasm" -c lowering.cubin > "$SRC/lowering.sm90.sass"
"$BIN/cu++filt" < "$SRC/lowering.sm90.sass" > "$SRC/lowering.sm90.demangled.sass"

"$BIN/ptxas" -arch=sm_90 -lineinfo --maxrregcount=24 "$SRC/lowering.sm90.ptx" -o rr24.cubin
"$BIN/nvdisasm" -c rr24.cubin > "$SRC/lowering.sm90.rr24.sass"
"$BIN/nvdisasm" -c --print-life-ranges --life-range-mode count rr24.cubin \
	> "$SRC/lowering.sm90.rr24.life.sass"

"$BIN/nvcc" -lineinfo -ptx -arch=sm_121a lowering.cu -o "$SRC/lowering.sm121a.ptx"
"$BIN/ptxas" -arch=sm_121a -lineinfo "$SRC/lowering.sm121a.ptx" -o lowering121a.cubin
"$BIN/nvdisasm" -c lowering121a.cubin > "$SRC/lowering.sm121a.sass"

"$BIN/nvcc" -lineinfo -ptx -arch=sm_90 other.cu -o "$SRC/other.sm90.ptx"
"$BIN/ptxas" -arch=sm_90 -lineinfo "$SRC/other.sm90.ptx" -o other.cubin
"$BIN/nvdisasm" -c other.cubin > "$SRC/other.sm90.sass"

"$BIN/nvcc" --version | tail -2
