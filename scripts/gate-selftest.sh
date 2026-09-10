#!/bin/sh
# Exercise the Makefile without changing the real workspace or its modules.
set -eu
mkdir -p "$HOME/tmp"
fixture=$(mktemp -d "$HOME/tmp/lib-gate-selftest.XXXXXX")
trap 'rm -rf "$fixture"' EXIT
trap 'exit 1' HUP INT TERM
cp Makefile "$fixture/Makefile"
cd "$fixture"
unset GOWORK GOFLAGS MAKEFLAGS MFLAGS
export GOTOOLCHAIN=local
git init -q

must_fail() {
	if make "$1" >output 2>&1; then
		cat output
		echo "FAIL: $1 succeeded: $2" >&2
		exit 1
	fi
	if ! grep -Fq "$3" output; then
		cat output
		echo "FAIL: $1 failed for an unexpected reason: $2" >&2
		exit 1
	fi
	echo "PASS: $1 rejects $2"
}

cat >go.work <<'WORK'
go 1.20
use (
	./missing
)
WORK
for target in build test; do
	must_fail "$target" 'missing workspace module' 'missing/go.mod'
done

mkdir bin
cat >bin/go <<'GO'
#!/bin/sh
# Simulate successful discovery with an empty result.
[ "$1" = list ] || exit 99
GO
chmod +x bin/go
real_path=$PATH
export PATH="$fixture/bin:$PATH"
for target in build test; do
	must_fail "$target" 'empty module discovery' 'error: no modules found'
done
export PATH=$real_path

mkdir member
cat >member/go.mod <<'MOD'
module example.com/member

go 1.20
MOD
printf 'package member\n' >member/member.go
cat >go.work <<'WORK'
go 1.20
use (
	./member
)
WORK
make all >output 2>&1 || { cat output; exit 1; }
grep -q '==> Building .*member' output
grep -q '==> Testing .*member' output
echo 'PASS: all checks, builds, and tests a valid untracked module'

mkdir unlisted
cp member/go.mod unlisted/go.mod
must_fail check 'untracked module outside workspace' 'unlisted is not listed in go.work'
must_fail all 'untracked module outside workspace' 'unlisted is not listed in go.work'
rm -r unlisted
printf 'package orphan\n' >orphan.go
must_fail check 'untracked Go file outside a module' 'orphan.go is not part of any module'
rm orphan.go

mkdir ignored
cp member/go.mod ignored/go.mod
printf 'ignored/\n' >.gitignore
make check >output 2>&1 || { cat output; exit 1; }
echo 'PASS: check excludes ignored output'
