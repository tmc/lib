# Manual gates

The live tests need `libtree-sitter` and a C grammar shared library. They
skip when either is missing, so a green run on a bare machine proves only
the generated layouts.

To run them on macOS with Homebrew's tree-sitter and Neovim:

```sh
P=$(brew --prefix)
TREE_SITTER_LIB_PATH=$P/lib/libtree-sitter.dylib \
TREE_SITTER_TEST_GRAMMAR=$P/lib/nvim/parser/c.so:c \
go test -count=1 -v ./...
```

`TREE_SITTER_TEST_GRAMMAR` is `path:name`, where `name` is the suffix of the
grammar's `tree_sitter_<name>` export. Without it the tests look for
`lib/nvim/parser/c.so` under `$HOMEBREW_PREFIX`, `/opt/homebrew`,
`/usr/local`, and `/usr`.

On Linux without packages, build both from source:

```sh
git clone --depth 1 --branch v0.26.11 https://github.com/tree-sitter/tree-sitter
(cd tree-sitter && make libtree-sitter.so)
git clone --depth 1 https://github.com/tree-sitter/tree-sitter-c
cc -shared -fPIC -O2 -Itree-sitter-c/src tree-sitter-c/src/parser.c -o c.so
TREE_SITTER_LIB_PATH=$PWD/tree-sitter/libtree-sitter.so \
TREE_SITTER_TEST_GRAMMAR=$PWD/c.so:c go test -count=1 -v ./...
```

Last verified 2026-09-01 with tree-sitter 0.26.11 and a C grammar at
ABI 15, all 16 tests passing:

| Target | Library | Grammar | Notes |
| --- | --- | --- | --- |
| macOS arm64 | Homebrew | Neovim `c.so` | with `-race` |
| Linux arm64 (Ubuntu, GB10) | built from source | built from source | cross-compiled test binary |

Linux amd64 is configured but has not been run.
