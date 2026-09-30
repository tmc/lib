#!/bin/sh
# Eject the pinned FFI registrar; pass -verify to check without writing.
#
# FFI_SRC names a tmc/ffi clone that has the pinned revision. The eject runs
# from a temporary checkout of that revision, so the clone may be on any
# branch and carry local changes.
set -eu
module=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
src=$(CDPATH= cd -- "${FFI_SRC:-$module/../../ffi}" && pwd)
revision=966f69c14c4e06a02708e3b1669fdfbafbeebf84
if ! git -C "$src" cat-file -e "$revision^{commit}" 2>/dev/null; then
    echo "ffi: $src does not have revision $revision; fetch it" >&2
    exit 1
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone -q --shared --no-checkout "$src" "$tmp/ffi"
git -C "$tmp/ffi" checkout -q --detach "$revision"
# Build rather than go run: only a build stamps the revision into the headers.
env GOWORK=off go -C "$tmp/ffi/cmd/ffi-eject" build -buildvcs=true -o "$tmp/ffi-eject" .
env GOWORK=off "$tmp/ffi-eject" \
    -src "$tmp/ffi" -mod "$module" -o "$module/internal/ffi" \
    -mode inline -pkg ffi -prefix nvidiaffi_ -with-errno -with-register \
    -purego-version v0.11.0 "$@"
