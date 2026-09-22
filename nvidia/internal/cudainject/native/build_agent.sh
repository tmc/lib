#!/bin/sh
# build_agent.sh compiles cupti_agent.c into the loadable CUPTI producer
# that cudaprof capture injects. It is not built by go build: the producer
# is C, links against the CUDA toolkit's CUPTI, and must be compiled on a
# machine that has one.
#
# Usage:
#
#	build_agent.sh [output directory]
#
# The default output directory is where go install puts binaries, which is
# one of the places cudaprof capture looks for the agent, so a build with
# no arguments is enough to make capture work without --inject.
#
# CUDA_HOME names the toolkit, defaulting to /usr/local/cuda. CC names the
# compiler.
set -eu

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${CUDA_HOME:=/usr/local/cuda}"
: "${CC:=cc}"

target=$(uname -m)-linux
if [ "$target" = aarch64-linux ]; then
	# The toolkit names the server-class arm64 target sbsa-linux.
	target=sbsa-linux
fi

out=${1:-}
if [ -z "$out" ]; then
	out=$(go env GOBIN)
fi
if [ -z "$out" ]; then
	out=$(go env GOPATH)/bin
fi
mkdir -p "$out"

# The revision is stamped into the agent's capability record, so a capture
# says which source produced it. A build outside a work tree leaves the
# field absent, which is honest; a wrong revision would not be.
rev=$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || true)
if [ -n "$rev" ] && ! git -C "$dir" diff --quiet 2>/dev/null; then
	rev="$rev-dirty"
fi
if [ -n "$rev" ]; then
	set -- -DCUDAPROF_AGENT_REVISION="\"$rev\""
else
	set --
fi

$CC -std=c11 -O2 -fPIC -shared "$@" \
	-I"$CUDA_HOME/targets/$target/include" -I"$CUDA_HOME/include" -I"$dir" \
	"$dir/cupti_agent.c" -o "$out/cupti_agent.so" \
	-L"$CUDA_HOME/targets/$target/lib" -L"$CUDA_HOME/lib64" -lcupti -lpthread

echo "built $out/cupti_agent.so${rev:+ from $rev}"
