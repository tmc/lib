#!/bin/sh
# Build every workspace module for the targets it claims in scripts/platforms.txt.
#
# The claim list is the point: source guarded by a build tag that no local build
# selects is invisible to every other gate in this repository, so a windows-only
# loader can stop compiling and nothing on a darwin workstation notices.
set -eu

list=scripts/platforms.txt
[ -f go.work ] || { echo "error: run from the repository root" >&2; exit 1; }
[ -f "$list" ] || { echo "error: $list is missing" >&2; exit 1; }

mkdir -p "$HOME/tmp"
work=$(mktemp -d "$HOME/tmp/lib-platform-check.XXXXXX")
trap 'rm -rf "$work"' EXIT
trap 'exit 1' HUP INT TERM

# Strip comments and blank lines once; every later pass reads this.
sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$list" >"$work/claims"

awk '{print $1}' "$work/claims" | sort >"$work/claimed"
sed -n 's|^[[:space:]]*\./\([^[:space:]]*\).*|\1|p' go.work | sort >"$work/used"
if ! cmp -s "$work/claimed" "$work/used"; then
	echo "error: $list does not name the same modules as go.work" >&2
	comm -23 "$work/claimed" "$work/used" | sed 's/^/  only in platforms.txt: /' >&2
	comm -13 "$work/claimed" "$work/used" | sed 's/^/  only in go.work:      /' >&2
	exit 1
fi

: >"$work/failures"
while IFS= read -r line; do
	dir=${line%%[	 ]*}
	for target in ${line#*[	 ]}; do
		os=${target%/*}
		arch=${target#*/}
		if (cd "$dir" && GOOS="$os" GOARCH="$arch" go build ./...) >"$work/build.log" 2>&1; then
			echo "ok   $dir $target"
		else
			echo "FAIL $dir $target"
			sed 's/^/     /' "$work/build.log"
			echo "$dir $target" >>"$work/failures"
		fi
	done
done <"$work/claims"

if [ -s "$work/failures" ]; then
	echo "error: claimed targets that do not build:" >&2
	sed 's/^/  /' "$work/failures" >&2
	exit 1
fi
echo "Every module builds for every platform it claims."
