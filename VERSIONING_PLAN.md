# Plan: per-subdirectory module versioning for tmc/lib

Status: in progress. `nvidia/go.mod`, `libp2p/go.mod`, and the repo-root
`go.work` exist in the working tree; nothing is tagged.

Standing constraints:

- **libp2p's dependency tree must never appear in the root module or any
  other module's go.mod/go.sum.** `libp2p/` is a module from birth
  (pinned go-libp2p v0.48.0 — v0.49 splits out a core module and causes
  ambiguous-import errors). The root go.mod requires only purego; keep it
  that way. Any future dep-heavy target likewise starts life as its own
  module.
- The repo is private and only lightly pushed; history may be reworked
  freely until first tags are published. That permits starting the public
  history clean with the multi-module layout in place from the beginning,
  and reorganizing directories (e.g. folding `simple-addition` into
  `examples/`, grouping the ccl family) before any tag exists.

## Why split

- Independent release cadence: `nvidia` iterates weekly; `fuse`/`kvm` are
  stable. One root module forces a shared version line and a shared
  `go.sum`.
- Dependency isolation is the immediate win: today `libp2p`'s ~90
  transitive dependencies (quic-go, pion/webrtc, prometheus, zap, …) land
  in the go.sum of every consumer of `github.com/tmc/lib`, including
  someone who only wants `kvm` (zero deps). Splitting makes every other
  module's dependency graph purego + x/sys at most.
- Semver honesty: generated binding surfaces change at different rates;
  per-module tags let a breaking regeneration bump only its own module.

## How Go submodule versioning works (the mechanics we rely on)

- A `go.mod` in a subdirectory makes that subtree its own module; the
  parent module automatically stops including it.
- The module `github.com/tmc/lib/nvidia` is versioned by git tags of the
  form `nvidia/v0.3.1` — tag prefix = directory path. Root module tags
  stay `v0.3.1`.
- The proxy serves each module from the tag whose prefix matches; tags are
  immutable once fetched (never retag, never delete — publish a new
  patch instead).
- `internal/` visibility follows directories, not modules, so a nested
  module could still import `github.com/tmc/lib/internal/...` — but that
  would create a dependency on the root module. Rule: **nested modules
  never import the root module.** (Nothing does today; keep it that way.
  `internal/packageplans` has no Go importers — it is generator input and
  stays with the root.)
- Cross-module imports become real versioned requires: `gonccl` will
  `require github.com/tmc/lib/nvidia v0.x.y` like any external dependency.
  Local development uses a `go.work` at the repo root so editing `nvidia`
  is immediately visible to `gonccl` without editing go.mod files.
  **No `replace` directives in committed go.mod files** — they break
  downstream consumers.

## Module layout

One module per top-level target; do not go finer (nvidia/driver,
nvidia/nccl etc. stay packages inside the `nvidia` module — they share a
release cadence and a generator).

Leaf modules (no tmc/lib deps):

| Module | Deps | Notes |
|---|---|---|
| `fuse` | none | |
| `kvm` | x/sys | |
| `nbd` | none | |
| `virtiofs` | none | |
| `vulkan` | purego | |
| `ggml` | purego | |
| `lldb` | purego | new; may want to bake before tagging |
| `libp2p` | go-libp2p tree | isolating this is the big go.sum win |
| `pccl` | purego | |
| `nvidia` | purego, x/sys | go.mod already drafted |
| `rocm` | purego | |

Dependent modules (require tagged tmc/lib modules):

| Module | Requires |
|---|---|
| `gopccl` | `pccl` |
| `gonccl` | `nvidia` |
| `gorccl` | `rocm` |
| `mgccl` | `gonccl`, `gorccl`, (`gopccl`?) — verify at split time |

**There is no root module.** Once the split completes, the root go.mod is
deleted: zero Go files live outside the target subdirs (`internal/
packageplans` and `docs/` are data/prose), so a root module would be
empty. Rootless is also fail-safe: with a root module, a new subdir
without its own go.mod silently joins it and ships under root tags; with
no root module, such a subdir is unimportable until someone consciously
creates its module — enforcing module-from-birth. The root keeps only
`go.work`, README, docs. Costs accepted: `go get github.com/tmc/lib`
errors (nothing importable there), pkg.go.dev renders per-module docs
from each module dir, and CI enumerates modules (required anyway).

Empirically validated (go1.27, simulated rootless repo + file:// proxy
consumer): consumer `go get <module>@<tag>` resolves cleanly with no root
module; `<dir>/vX.Y.Z` tags map exactly to proxy version lists; a
go.mod-less subdir is fully inert (nothing silently joins). Maintainer-
side caveats found:

- `go build ./...` from the repo root **never works** in a rootless
  workspace (`directory prefix . does not contain modules listed in
  go.work`). The canonical build-everything invocation is per-module:
  `go list -m -f '{{.Dir}}'` and loop. Provide it as a Makefile target.
- `go work sync` needs every `require`d version fetchable from the proxy
  even for `use`d modules — run it only after tags are published, and
  expect it to reorder go.work.
- Dependent modules (gonccl→nvidia) must commit their own go.sum and
  refresh it on every dependency tag bump; outside the workspace they
  build only against published tags. That is the reproducibility working
  as intended.
- README note needed: `go get github.com/tmc/lib@latest` fails by design;
  point users at subpaths.
- CI should assert every .go file lives under a module listed in go.work
  — go.mod-less dirs are invisible, hence silently untested.

`examples/` becomes its own **untagged** module (`examples/go.mod`) so
example code can depend on everything without its dependencies or API
appearing in any released module. `simple-addition` (stray root dir) gets
foldered into examples.

## Versioning policy

- Everything starts at `v0.1.0`. Stay in v0 until a surface is
  deliberately frozen; v0 permits breaking changes while still giving
  consumers reproducible builds.
- Generated-code reality: a regeneration that removes or retypes an
  exported symbol is breaking. In v0 that is legal but should still bump
  minor (v0.2.0), with fixes bumping patch. Gate tags on
  `gorelease`-style API diff review (see tooling) rather than intuition.
- Promotion to v1 is per-module and deliberate (fuse/nbd/virtiofs are the
  plausible first candidates; nvidia stays v0 while the generator churns).
- Never tag v2+ without adding the `/v2` module-path suffix — for a
  submodule that means `module github.com/tmc/lib/nvidia/v2` and tags
  `nvidia/v2.0.0`. Avoid needing this by staying in v0/v1.

## Migration sequence

1. **Commit the in-flight state** — current worktree has modified root
   go.mod/go.sum and several untracked dirs (libp2p/, vulkan/,
   nvidia/go.mod...). Land those first; the split starts from a clean tree.
2. **Split leaf modules** — one commit per module: add `<dir>/go.mod`
   (module path `github.com/tmc/lib/<dir>`, `go` directive matching root),
   `go mod tidy` inside it, remove its requires from root go.mod, tidy
   root. Add repo-root `go.work` listing all modules (commit it — this is
   a self-contained monorepo, and a committed go.work keeps `go build
   ./...` at root working for every checkout; consumers ignore it).
3. **CI/test adjustment** — `go test ./...` no longer crosses module
   boundaries from the root; script it as a loop over `go work edit
   -json`-listed modules (or `git ls-files '**/go.mod'`).
4. **First tag wave** — tag every leaf module `<dir>/v0.1.0` on the same
   commit. Then convert dependent modules (gopccl/gonccl/gorccl): add
   go.mod requiring the just-tagged versions, tidy, commit, tag their
   v0.1.0. Then mgccl. Root module tags `v0.1.0` last (or is left
   untagged if nothing consumes it directly).
5. **Delete the root go.mod/go.sum** — the final split commit removes the
   root module entirely (see "There is no root module" above) and updates
   go.work to drop the `.` entry. No bare `vX.Y.Z` tags are ever created.

## Release workflow (steady state)

- Change in a leaf module: commit, `git tag <dir>/v0.x.y`, push tag.
- Change in `nvidia` that `gonccl` needs: tag `nvidia/v0.x.y` first, then
  bump gonccl's require, tidy, commit, tag `gonccl/v0.x.y`. Dependency
  order is enforced by the proxy — a dependent cannot require an untagged
  version (pseudo-versions work in a pinch but avoid them in tags).
- Add a small `release.sh` (or Makefile target) that, given a module dir:
  runs the module's tests, runs API-diff against the last tag, computes
  the next version, and prints the `git tag` command. Keep tagging itself
  manual.

## Tooling

- `golang.org/x/exp/cmd/gorelease` (API compatibility + suggested next
  version) as the pre-tag gate. If it proves too stale, `apidiff`
  directly.
- `go work sync` after cross-module require bumps.
- CI matrix: per-module test jobs; a repo-wide job that runs `go build`
  in every module with the go.work file active (catches cross-module
  breakage before tagging).

## Hazards

- **Tags are forever**: the module proxy caches immutably. A bad tag is
  fixed by a new patch tag, never by retagging.
- **Nested-module shadowing**: once `nvidia/go.mod` exists, root-module
  consumers on older root tags still see nvidia packages from those old
  tags. Cut the split and the tag wave close together so no root tag
  ships a half-split tree.
- **`GOFLAGS=-mod=mod` / stale GOPATH-mode assumptions**: this repo lives
  under GOPATH src; module mode ignores that, but scripts that assume one
  module root (`go list ./...` from root) need the go.work loop.
- **go.sum churn in dependent modules**: gonccl's go.sum will pin nvidia's
  hash per version; regenerating nvidia without tagging leaves gonccl
  unbuildable outside go.work. That is by design — it is what makes
  releases reproducible.
- **Never hand-write go.sum files** (repo policy); all sums come from
  `go mod tidy`.
