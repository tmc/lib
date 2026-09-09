#!/bin/sh
# Package a capture with portable viewers and the source needed to rebuild them.
set -eu
if [ "$#" -ne 2 ]; then
    echo "usage: $0 capture-directory new-output-directory" >&2
    exit 2
fi
module=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
capture=$(CDPATH='' cd -- "$1" && pwd)
mkdir -- "$2"
out=$(CDPATH='' cd -- "$2" && pwd)
case "$out/" in
    "$module/"*) echo "output must be outside the source module" >&2; exit 1 ;;
esac
if [ -e "$out.tar.gz" ]; then
    echo "$out.tar.gz already exists" >&2
    exit 1
fi
for name in main.go capture.go workload.go kernels.cu; do
    cmp "$capture/$name" "$module/examples/ptx-sass-trace/$name"
done
cp -R "$capture/." "$out/"
mkdir "$out/bin" "$out/src"
(cd "$module" && COPYFILE_DISABLE=1 tar --no-xattrs --exclude=.git --exclude=vendor -cf "$out/source.tar" .)
tar -xf "$out/source.tar" -C "$out/src"
rm "$out/source.tar"
cd "$out/src"
export GOWORK=off
go mod vendor
for target in darwin-arm64 darwin-amd64 linux-arm64 linux-amd64; do
    GOOS=${target%-*} GOARCH=${target#*-} CGO_ENABLED=0 \
        go build -mod=vendor -trimpath -o "$out/bin/cudaprof-$target" ./cmd/cudaprof
done
cat > "$out/view" <<'VIEW'
#!/bin/sh
set -eu
dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
case $(uname -s) in
    Darwin) platform=darwin ;;
    Linux) platform=linux ;;
    *) echo "supported platforms: macOS and Linux" >&2; exit 1 ;;
esac
case $(uname -m) in
    arm64|aarch64) arch=arm64 ;;
    x86_64|amd64) arch=amd64 ;;
    *) echo "supported architectures: arm64 and amd64" >&2; exit 1 ;;
esac
if [ "$#" -eq 0 ]; then
    set -- -http=127.0.0.1:
fi
exec "$dir/bin/cudaprof-$platform-$arch" "$@" "$dir/capture.pprof"
VIEW
cat > "$out/trace" <<'TRACE'
#!/bin/sh
set -eu
dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
GOTOOLCHAIN=$(sed -n 's/.*"go_version": "\([^"]*\)".*/\1/p' "$dir/manifest.json")
export GOTOOLCHAIN
exec go tool trace "$@" "$dir/go.trace"
TRACE
chmod +x "$out/view" "$out/trace"
cat > "$out/src/BUILD.md" <<'BUILD'
# Build and capture

This directory is a snapshot of the NVIDIA Go module used for this capture.
Dependencies and their license files are in vendor. Go 1.26.2 or later is
required; use the producer's version in ../manifest.json to match its toolchain.

Build the viewer without downloading dependencies, from this directory:

```sh
GOWORK=off GOPROXY=off GOSUMDB=off CGO_ENABLED=0 \
  go build -mod=vendor -trimpath -o ../cudaprof ./cmd/cudaprof
../cudaprof -http=127.0.0.1:8087 ../gpu.pprof
```

Capture again on an NVIDIA host with the CUDA driver, NVRTC, and a matching
nvdisasm installed:

```sh
GOWORK=off GOPROXY=off GOSUMDB=off go run -mod=vendor ./examples/ptx-sass-trace \
  -nvdisasm /usr/local/cuda-13/bin/nvdisasm
```

The captured CUBIN targets the recorded GPU architecture. Viewing the recording
does not execute it. A fresh capture compiles for the current device.
BUILD
"$out/view" -top > "$out/profile.txt"
"$out/view" -disasm='compute|vector_add' > "$out/disassembly.txt"
"$out/view" -weblist='compute|vector_add' -output="$out/source.html"
cd "$(dirname -- "$out")"
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$out.tar.gz" "$(basename -- "$out")"
echo "$out.tar.gz"
