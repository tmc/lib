#!/bin/sh
# Eject the pinned FFI registrar; pass -verify to check without writing.
set -eu
module=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
src=$(CDPATH= cd -- "${FFI_SRC:-$module/../../ffi}" && pwd)
revision=750d19d5e3f6c0fc2b52a9acee444bc104c92265
actual=$(git -C "$src" rev-parse HEAD)
if [ "$actual" != "$revision" ] || [ -n "$(git -C "$src" status --porcelain)" ]; then
    echo "ffi: FFI_SRC must be a clean checkout of $revision (got $actual)" >&2
    exit 1
fi
exec env GOWORK=off go -C "$src/cmd/ffi-eject" run . \
    -src "$src" -mod "$module" -o "$module/internal/ffi" \
    -mode inline -pkg ffi -prefix nvidiaffi_ -with-register \
    -purego-version v0.11.0 "$@"
