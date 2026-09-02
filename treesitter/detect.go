package treesitter

import (
	"path/filepath"
	"strings"
	"sync"
)

// DetectLanguage guesses the grammar name for a file from its name, using
// the extensions and well-known file names the common grammars declare.
// It returns "" when the name is not recognized. tree-sitter itself does
// not identify languages; this table mirrors the file-types lists of the
// grammars' tree-sitter.json files, so a hit is a name [FindLanguage] can
// look up, not a promise that the grammar is installed.
//
// Ambiguous extensions resolve to the most common grammar: .h is c, .m is
// objc, .scm is scheme, .v is verilog.
func DetectLanguage(filename string) string {
	base := filepath.Base(filename)
	if lang, ok := languageByFile[base]; ok {
		return lang
	}
	ext := strings.ToLower(filepath.Ext(base))
	if ext == "" {
		return ""
	}
	return languageByExt[ext]
}

// languageByFile maps exact file names to grammar names.
var languageByFile = map[string]string{
	"go.mod":         "gomod",
	"go.sum":         "gosum",
	"go.work":        "gowork",
	"Makefile":       "make",
	"GNUmakefile":    "make",
	"makefile":       "make",
	"CMakeLists.txt": "cmake",
	"Dockerfile":     "dockerfile",
	"Containerfile":  "dockerfile",
	"Gemfile":        "ruby",
	"Rakefile":       "ruby",
	"BUILD":          "starlark",
	"BUILD.bazel":    "starlark",
	"WORKSPACE":      "starlark",
	"MODULE.bazel":   "starlark",
	"meson.build":    "meson",
	".gitignore":     "gitignore",
	".gitattributes": "gitattributes",
	"COMMIT_EDITMSG": "gitcommit",
	".vimrc":         "vim",
	"vimrc":          "vim",
	"init.vim":       "vim",
	".zshrc":         "bash",
	".bashrc":        "bash",
	".bash_profile":  "bash",
	".profile":       "bash",
}

// languageByExt maps lower-case extensions, including the dot, to grammar
// names.
var languageByExt = map[string]string{
	".c": "c", ".h": "c",
	".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".c++": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp", ".h++": "cpp", ".inl": "cpp", ".ipp": "cpp",
	".m": "objc", ".mm": "objcpp",
	".cu": "cuda", ".cuh": "cuda",
	".go": "go",
	".rs": "rust",
	".py": "python", ".pyi": "python", ".pyw": "python",
	".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
	".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
	".tsx":  "tsx",
	".java": "java",
	".kt":   "kotlin", ".kts": "kotlin",
	".swift": "swift",
	".scala": "scala", ".sc": "scala",
	".cs": "c_sharp",
	".fs": "fsharp", ".fsi": "fsharp", ".fsx": "fsharp",
	".rb": "ruby", ".rake": "ruby", ".gemspec": "ruby",
	".php": "php", ".phtml": "php",
	".lua": "lua",
	".sh":  "bash", ".bash": "bash", ".zsh": "bash", ".ksh": "bash",
	".fish": "fish",
	".ps1":  "powershell", ".psm1": "powershell", ".psd1": "powershell",
	".pl": "perl", ".pm": "perl",
	".r":  "r",
	".jl": "julia",
	".hs": "haskell", ".lhs": "haskell",
	".ml": "ocaml", ".mli": "ocaml_interface",
	".ex": "elixir", ".exs": "elixir",
	".erl": "erlang", ".hrl": "erlang",
	".elm": "elm",
	".clj": "clojure", ".cljs": "clojure", ".cljc": "clojure", ".edn": "clojure",
	".lisp": "commonlisp", ".cl": "commonlisp", ".el": "elisp",
	".scm": "scheme", ".ss": "scheme", ".rkt": "racket",
	".zig":  "zig",
	".nix":  "nix",
	".dart": "dart",
	".d":    "d",
	".nim":  "nim",
	".odin": "odin",
	".sol":  "solidity",
	".json": "json", ".jsonc": "jsonc", ".json5": "json5",
	".yaml": "yaml", ".yml": "yaml",
	".toml": "toml",
	".ini":  "ini", ".cfg": "ini",
	".xml": "xml", ".xsd": "xml", ".xsl": "xml", ".svg": "xml", ".plist": "xml",
	".html": "html", ".htm": "html",
	".css": "css", ".scss": "scss",
	".vue": "vue", ".svelte": "svelte", ".astro": "astro",
	".md": "markdown", ".markdown": "markdown",
	".rst": "rst",
	".org": "org",
	".tex": "latex", ".sty": "latex", ".cls": "latex", ".bib": "bibtex",
	".sql":     "sql",
	".graphql": "graphql", ".gql": "graphql",
	".proto":  "proto",
	".thrift": "thrift",
	".capnp":  "capnp",
	".hcl":    "hcl", ".tf": "terraform", ".tfvars": "terraform",
	".mk":    "make",
	".cmake": "cmake",
	".ninja": "ninja",
	".bzl":   "starlark", ".star": "starlark",
	".vim":  "vim",
	".diff": "diff", ".patch": "diff",
	".csv": "csv", ".tsv": "tsv",
	".glsl": "glsl", ".vert": "glsl", ".frag": "glsl", ".geom": "glsl", ".comp": "glsl",
	".hlsl": "hlsl", ".wgsl": "wgsl", ".metal": "cpp",
	".v": "verilog", ".sv": "verilog", ".svh": "verilog",
	".vhd": "vhdl", ".vhdl": "vhdl",
	".f": "fortran", ".f90": "fortran", ".f95": "fortran", ".f03": "fortran", ".f08": "fortran",
	".groovy": "groovy", ".gradle": "groovy",
	".s": "asm", ".asm": "asm",
	".wat": "wast",
	".ll":  "llvm",
	".re":  "regex",
}

var foundLanguages sync.Map // name -> *Language
