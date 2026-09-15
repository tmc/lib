#!/bin/sh
# Pre-tag API gate: compare the working tree's exported API against a released
# tag and refuse to pass when the difference is incompatible.
#
# apidiff always exits 0, so the verdict here is derived from its output rather
# than its status. The comparison runs once per target platform: the exported
# surface is build-tag dependent, and a darwin-only run does not see the
# linux-only CUDA packages at all.
#
# Usage: sh scripts/apidiff.sh [baseline-tag]

set -eu

BASELINE=${1:-${BASELINE:-}}
PLATFORMS=${PLATFORMS:-"linux/amd64 darwin/arm64"}
MODULE=github.com/tmc/lib/nvidia

command -v apidiff >/dev/null 2>&1 || {
	echo "apidiff: not installed; go install golang.org/x/exp/cmd/apidiff@latest" >&2
	exit 1
}

root=$(git rev-parse --show-toplevel)
if [ -z "$BASELINE" ]; then
	BASELINE=$(git -C "$root" tag --list 'nvidia/v*' --sort=-v:refname | head -1)
fi
[ -n "$BASELINE" ] || { echo "apidiff: no nvidia/v* tag to compare against" >&2; exit 1; }

mkdir -p "$HOME/tmp"
work=$(mktemp -d "$HOME/tmp/nvidia-apidiff.XXXXXX")
base="$work/baseline"
trap 'git -C "$root" worktree remove --force "$base" >/dev/null 2>&1 || :; rm -rf "$work"' EXIT INT TERM HUP

git -C "$root" worktree add -q --detach "$base" "$BASELINE"

echo "==> apidiff against $BASELINE"
rc=0
for p in $PLATFORMS; do
	goos=${p%/*}
	goarch=${p#*/}
	old="$work/old.$goos-$goarch.api"
	new="$work/new.$goos-$goarch.api"

	(cd "$base/nvidia" && GOWORK=off GOOS="$goos" GOARCH="$goarch" apidiff -m -w "$old" "$MODULE")
	(cd "$root/nvidia" && GOWORK=off GOOS="$goos" GOARCH="$goarch" apidiff -m -w "$new" "$MODULE")

	out="$work/diff.$goos-$goarch.txt"
	GOWORK=off GOOS="$goos" GOARCH="$goarch" apidiff -m -incompatible "$old" "$new" >"$out" 2>&1

	# apidiff reports skipped internal packages on stdout alongside findings;
	# only the "- "/"+ " lines are changes.
	n=$(awk '/^[-+] /{c++} END{print c+0}' "$out")
	echo "--- $goos/$goarch: $n incompatible change(s)"
	if [ "$n" -gt 0 ]; then
		grep '^[-+] ' "$out" || :
		rc=1
	fi
done

if [ "$rc" -ne 0 ]; then
	echo
	if [ "${ALLOW_INCOMPATIBLE:-0}" = "1" ]; then
		echo "apidiff: incompatible changes since $BASELINE, recorded as intended (ALLOW_INCOMPATIBLE=1)."
		exit 0
	fi
	echo "apidiff: incompatible changes since $BASELINE."
	echo "Tag a new major/minor accordingly, or set ALLOW_INCOMPATIBLE=1 to record them as intended."
	exit 1
fi
echo "apidiff: no incompatible changes since $BASELINE."
