# Testing trainstat

`TestScript` builds trainstat and benchstat, then runs a small Go module that
imports this module's trainstat package. Its `gotest` script uses `GOPROXY=off`
and `GOSUMDB=off`; the module cache must already contain the dependencies pinned
in `nvidia/go.mod` and `nvidia/go.sum`.

From the `nvidia` module, provision its four required modules in a disposable
module while the configured proxy is available, then run the package offline:

```sh
work="$HOME/tmp/lib/$(date +%Y%m%d)-trainstat"
mkdir -p "$work/tmp" "$work/cache/build" "$work/cache/modules" "$work/provision"
export TMPDIR="$work/tmp" GOCACHE="$work/cache/build" GOMODCACHE="$work/cache/modules"
export GOWORK=off
go list -mod=readonly -m -f '{{.Path}}@{{.Version}}' \
    github.com/aclements/go-moremath golang.org/x/perf golang.org/x/tools rsc.io/script \
    >"$work/modules"
(
    cd "$work/provision" || exit
    test -f go.mod || go mod init validation.invalid/trainstat-cache
    while IFS= read -r module; do
        go mod edit "-require=$module" || exit
    done <"$work/modules"
    while IFS= read -r module; do
        go mod download "$module" || exit
    done <"$work/modules"
    go mod verify
) || exit
GOPROXY=off GOSUMDB=off go test -mod=readonly -count=1 ./x/cmd/trainstat
```

The script environment preserves the caller's Go cache, proxy, checksum and
toolchain settings. It keeps HOME unchanged. Each script has its own temporary
directory. An empty offline cache is a prerequisite failure, not a reason to
skip the script or fetch modules during the offline run.
