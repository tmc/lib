#!/bin/sh
# Regenerates the kernels.* fixtures in this directory from the example's own
# kernels.cu, the translation unit the example compiles at run time. Run on a
# host with CUDA installed; set CUDA to its prefix. The build happens in a
# fixed directory so that the absolute path the disassembler embeds in its
# "//## File" comments is reproducible across hosts:
#
#	CUDA=/usr/local/cuda-13.4 ./gen.sh
#
# The generated files are byte-exact toolchain output. Do not edit them by
# hand: a fixture edited to agree with the loader proves nothing about what the
# toolchain emits.
#
# kernels.cu is copied rather than authored here, so that the package's
# testdata is self-contained under go:embed while the unit has one author. The
# nvdisasm options below are the ones capture.go passes, so the fixture is the
# disassembly this example really writes into a bundle; the example compiles
# through NVRTC for the device it finds, which is the one difference.
#
# padding.{ptx,sass} and ds4_math.cu are a different fixture with its own
# provenance; see padding.README. This script does not touch them.
set -e
: "${CUDA:=/usr/local/cuda}"
BIN="$CUDA/bin"
BUILD=/tmp/ptx-sass-trace-fixtures
SRC=$(cd "$(dirname "$0")" && pwd)

cp "$SRC/../kernels.cu" "$SRC/kernels.cu"

rm -rf "$BUILD"
mkdir -p "$BUILD"
cp "$SRC/kernels.cu" "$BUILD"
cd "$BUILD"

"$BIN/nvcc" -lineinfo -ptx -arch=sm_90 kernels.cu -o "$SRC/kernels.ptx"
"$BIN/ptxas" -arch=sm_90 -lineinfo "$SRC/kernels.ptx" -o kernels.cubin
"$BIN/nvdisasm" --print-code --print-line-info-inline --print-line-info-ptx \
	kernels.cubin > "$SRC/kernels.sass"
"$BIN/nvdisasm" --output-control-flow-graph --print-instr-offsets-cfg \
	kernels.cubin > "$SRC/kernels.cfg.dot"
"$BIN/nvdisasm" --output-control-flow-graph-with-basic-blocks --print-instr-offsets-cfg \
	kernels.cubin > "$SRC/kernels.bbcfg.dot"
xxd -p -c 1000000 kernels.cubin > "$SRC/kernels.cubin.hex"

"$BIN/nvcc" --version | tail -2
