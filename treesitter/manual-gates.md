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
ABI 15, all 35 tests passing:

| Target | Library | Grammar | Notes |
| --- | --- | --- | --- |
| macOS arm64 | Homebrew | Neovim `c.so` | with `-race` |
| Linux arm64 (Ubuntu, GB10) | built from source | built from source | cross-compiled test binary |

Linux amd64 is configured but has not been run.

## Native memory

`TestCallbackRegistryDrains` proves the wrapper releases every Go value it
hands to C, but it says nothing about what the C allocator does. Upstream
reports memory that is not returned to the system after large parses
(tree-sitter/go-tree-sitter#36, #52), so measure resident set size
directly rather than inferring it from `Close` coverage.

The reproducer parses one 4 MB C file repeatedly, closing each tree, and
samples RSS through `ps` after a `runtime.GC`:

```go
src := big4MBCSource()
p, _ := treesitter.NewParser()
p.SetLanguage(lang)
for i := 1; i <= 300; i++ {
	tree, err := p.Parse(src, nil)
	if err != nil {
		log.Fatal(err)
	}
	tree.Close()
	if i%50 == 0 {
		runtime.GC()
		fmt.Println(i, rssKiB())
	}
}
```

Measured 2026-09-01 on macOS arm64 (M4 Max), tree-sitter 0.26.11, C
grammar at ABI 15:

| Parses | RSS (KiB) |
| --- | --- |
| 0 (baseline) | 26,080 |
| 50 | 254,432 |
| 100 | 254,560 |
| 200 | 254,736 |
| 300 | 254,832 |

The high-water mark is reached within the first parses and is about 60
times the source size; parsing is memory-hungry and `malloc` does not
return the arena to the operating system. Growth after that is 400 KiB
over 250 parses, roughly 1.6 KiB per parse, which is allocator noise
rather than the linear growth #52 describes. Repeat this measurement on
Linux before claiming the behavior is portable.

## Callback cost

`ParseContext` adds a progress callback and `ParseReaderAt` adds a read
callback for every window, so both cross from C into Go during a parse.
Measured on the same machine over one 64 KiB source, `-benchtime 200x
-count 3`:

| Benchmark | ns/op | Relative |
| --- | --- | --- |
| `Parse` | 2,177,119 | — |
| `ParseContext` | 2,211,498 | +1.6% |
| `ParseReaderAt` | 2,192,355 | +0.8% |

Rerun with `go test -run '^$' -bench . -benchtime 200x -count 3` before
quoting these numbers on other hardware.

## Installing grammars

`TestInstall` in `treesitter/grammars` downloads pinned source archives and
compiles them, so it is off by default:

```sh
TREE_SITTER_LIB_PATH=$P/lib/libtree-sitter.dylib \
TREE_SITTER_INSTALL_TEST=1 go test -count=1 -v ./grammars
```

It builds json, go, and typescript, checks that a second build reuses the
library rather than rebuilding it, parses through `FindLanguage` to prove
the install path and the search path agree, and checks that a wrong
checksum is refused. Last verified 2026-09-01 on macOS arm64 with Apple
clang: three grammars downloaded and compiled in 3.4s.
