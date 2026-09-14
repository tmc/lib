#!/bin/sh
# Compile the test packages behind every non-standard build tag in the tree.
#
# Hardware qualification tests sit behind tags like cupti_physical, so nothing
# a workstation runs ever compiles them: they can stop building for months and
# every gate stays green. This builds each one without running it, which is the
# most a machine with no GPU can honestly check.
#
# The tag list is discovered, not declared, so a new tag is covered the day it
# appears.
set -eu

[ -f go.work ] || { echo "error: run from the repository root" >&2; exit 1; }

mkdir -p "$HOME/tmp"
work=$(mktemp -d "$HOME/tmp/lib-testtag-check.XXXXXX")
trap 'rm -rf "$work"' EXIT
trap 'exit 1' HUP INT TERM

# Every //go:build term in a test file that is not a GOOS, GOARCH, or one of
# the toolchain's own terms.
known='aix|android|darwin|dragonfly|freebsd|hurd|illumos|ios|js|linux|netbsd|openbsd|plan9|solaris|wasip1|windows|unix|386|amd64|arm|arm64|loong64|mips|mips64|mips64le|mipsle|ppc64|ppc64le|riscv64|s390x|wasm|cgo|race|msan|asan|gc|gccgo|go1\.[0-9]+|ignore'
git ls-files '*_test.go' |
	while IFS= read -r f; do
		sed -n 's|^//go:build ||p' "$f" | sed "s|^|$f\t|"
	done >"$work/constraints"
awk -F'\t' '{print $2}' "$work/constraints" | tr -c 'A-Za-z0-9_.' '\n' |
	grep -v '^$' | sort -u | grep -vxE "$known" >"$work/tags"

[ -s "$work/tags" ] || { echo "No tagged test files."; exit 0; }

# scripts/platforms.txt says which GOOS each module claims. A tag whose GOOS
# the module does not claim is skipped rather than failed: the module never
# promised to compile there at all.
sed -e 's/#.*//' -e '/^[[:space:]]*$/d' scripts/platforms.txt >"$work/claims"

: >"$work/failures"
while IFS= read -r tag; do
	# The packages that actually carry the tag; building the whole module for
	# each tag would cost minutes and check nothing more. Only un-negated
	# occurrences count: llama-infer names tpu far more often as "!tpu", and
	# those are the files selecting the tag takes away.
	grep -E "(^|[^A-Za-z0-9_!])$tag([^A-Za-z0-9_]|$)" "$work/constraints" >"$work/lines"
	awk -F'\t' '{print $1}' "$work/lines" | sed 's|/[^/]*$||' | sort -u >"$work/dirs"
	while IFS= read -r dir; do
		mod=$dir
		while [ "$mod" != "." ] && [ ! -f "$mod/go.mod" ]; do mod=$(dirname "$mod"); done
		[ "$mod" != "." ] || continue
		rel=$(printf '%s\n' "$dir" | sed "s|^$mod||; s|^/||")
		[ -n "$rel" ] || rel=.

		# A tag can be meaningful on one GOOS only: llama-infer's tpu tests
		# are "linux && tpu", and selecting the tag on any other GOOS drops
		# the files the rest of the package needs. Take the GOOS from the
		# constraints themselves rather than guessing.
		goos=$(awk -F'\t' -v d="$dir/" '$1 ~ "^"d {print $2}' "$work/lines" |
			tr -c 'A-Za-z0-9_' '\n' |
			grep -xE 'darwin|linux|windows' | sort -u)
		case $(printf '%s' "$goos" | wc -w | tr -d ' ') in
		0) goos=$(go env GOOS) ;;
		1) ;;
		*) goos=$(go env GOOS) ;;
		esac
		if ! grep -qE "^$mod	.*[ 	]$goos/" "$work/claims"; then
			echo "skip $tag $dir ($mod does not claim $goos)"
			continue
		fi
		# The workspace resolves the modules that still carry replacements,
		# the same way "make build" and "make test" do.
		if (cd "$mod" && GOOS="$goos" go test -tags "$tag" -run '^$' -count=1 "./$rel") >"$work/out" 2>&1; then
			echo "ok   $tag $dir ($goos)"
		else
			echo "FAIL $tag $dir ($goos)"
			sed 's/^/     /' "$work/out"
			echo "$tag $dir $goos" >>"$work/failures"
		fi
	done <"$work/dirs"
done <"$work/tags"

if [ -s "$work/failures" ]; then
	echo "error: tagged test packages that do not build:" >&2
	sed 's/^/  /' "$work/failures" >&2
	exit 1
fi
echo "Every tagged test package builds."
