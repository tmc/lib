#!/bin/sh
set -eu
mkdir -p "$HOME/tmp"
fixture=$(mktemp -d "$HOME/tmp/lib-release-selftest.XXXXXX")
trap 'rm -rf "$fixture"' EXIT
trap 'exit 1' HUP INT TERM
cp scripts/check-release.py "$fixture/check-release.py"
cd "$fixture"
unset GOWORK GOFLAGS
export GOTOOLCHAIN=local
mkdir member
printf 'go 1.20\nuse ./member\n' >go.work
printf 'module github.com/tmc/lib/member\ngo 1.20\n' >member/go.mod
printf 'Fixture license\n' >LICENSE
python3 check-release.py >output 2>&1

echo 'PASS: valid release metadata'
must_fail() {
    if python3 check-release.py >output 2>&1; then
        echo "FAIL: accepted $1" >&2
        exit 1
    fi
    grep -Fq "$1" output || { cat output; exit 1; }
    echo "PASS: rejects $1"
}
rm LICENSE
must_fail 'repository license is missing'
printf 'Fixture license\n' >LICENSE
printf '\nreplace example.com/dep => ../dep\n' >>member/go.mod
must_fail 'remove replacement'
printf 'module github.com/tmc/lib/member\ngo 1.20\nrequire example.com/dep v0.0.0\n' >member/go.mod
must_fail 'placeholder version'
printf 'module example.com/wrong\ngo 1.20\n' >member/go.mod
must_fail 'module path must be'

printf 'go 1.20\n' >go.work
must_fail 'workspace has no modules'
printf 'go 1.20\nuse ../external\n' >go.work
must_fail 'workspace member must be inside the repository'
