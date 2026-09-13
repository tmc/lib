#!/bin/sh
# Regenerates the .ptx fixtures in this directory from the .cu sources.
#
# Run on a host with CUDA installed; set CUDA to its prefix. The build happens
# in a fixed directory so that the absolute path NVVM embeds in .file is
# reproducible across hosts:
#
#	CUDA=/usr/local/cuda-13.0 ./gen.sh
#
# The generated .ptx files are byte-exact compiler output. Do not edit them by
# hand -- a fixture edited to agree with the parser proves nothing. Each file
# keeps the NVVM header comment naming the CUDA release that produced it.
set -e
: "${CUDA:=/usr/local/cuda}"
NVCC="$CUDA/bin/nvcc"
BUILD=/tmp/cudaptx-fixtures
SRC=$(cd "$(dirname "$0")" && pwd)

rm -rf "$BUILD"
mkdir -p "$BUILD"
cp "$SRC"/*.cu "$BUILD"
cd "$BUILD"

"$NVCC" -ptx -arch=sm_90 minimal.cu -o "$SRC/minimal.sm90.ptx"
"$NVCC" -lineinfo -rdc=true -ptx -arch=sm_90 lexer.cu -o "$SRC/lexer.sm90.ptx"
"$NVCC" -lineinfo -rdc=true -ptx -arch=sm_121a lexer.cu -o "$SRC/lexer.sm121a.ptx"
"$NVCC" -lineinfo -ptx -arch=sm_90 vector.cu -o "$SRC/vector.sm90.ptx"

"$NVCC" --version | tail -2
