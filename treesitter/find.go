package treesitter

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// GrammarPathEnv names the environment variable holding extra directories
// for [FindLanguage], separated by the OS path list separator.
const GrammarPathEnv = "TREE_SITTER_GRAMMAR_PATH"

// GrammarDirs returns the directories FindLanguage searches, in order:
// the entries of TREE_SITTER_GRAMMAR_PATH, TREE_SITTER_LIBDIR (the
// tree-sitter CLI's setting) or the CLI's default cache directory, the
// parser directories of Neovim and Emacs installations, and the common
// system library directories. Directories that do not exist are omitted.
func GrammarDirs() []string {
	var dirs []string
	for _, d := range filepath.SplitList(os.Getenv(GrammarPathEnv)) {
		if d != "" {
			dirs = append(dirs, d)
		}
	}
	if d := os.Getenv("TREE_SITTER_LIBDIR"); d != "" {
		dirs = append(dirs, d)
	} else if cache, err := os.UserCacheDir(); err == nil {
		dirs = append(dirs, filepath.Join(cache, "tree-sitter", "lib"))
	}
	if home, err := os.UserHomeDir(); err == nil {
		data := os.Getenv("XDG_DATA_HOME")
		if data == "" {
			data = filepath.Join(home, ".local", "share")
		}
		dirs = append(dirs,
			filepath.Join(data, "nvim", "lazy", "nvim-treesitter", "parser"),
			filepath.Join(data, "nvim", "site", "parser"),
			filepath.Join(home, ".emacs.d", "tree-sitter"),
			filepath.Join(home, ".config", "emacs", "tree-sitter"),
		)
	}
	if nvim, err := exec.LookPath("nvim"); err == nil {
		if real, err := filepath.EvalSymlinks(nvim); err == nil {
			dirs = append(dirs, filepath.Join(filepath.Dir(real), "..", "lib", "nvim", "parser"))
		}
	}
	prefixes := []string{os.Getenv("HOMEBREW_PREFIX"), "/opt/homebrew", "/usr/local", "/usr"}
	for _, prefix := range prefixes {
		if prefix == "" {
			continue
		}
		dirs = append(dirs, filepath.Join(prefix, "lib", "nvim", "parser"))
	}
	dirs = append(dirs, "/usr/local/lib", "/usr/lib64", "/usr/lib", "/usr/lib/tree-sitter", "/usr/lib/"+runtime.GOARCH+"-linux-gnu")

	seen := make(map[string]bool)
	out := dirs[:0]
	for _, d := range dirs {
		d = filepath.Clean(d)
		if seen[d] {
			continue
		}
		seen[d] = true
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			out = append(out, d)
		}
	}
	return out
}

// GrammarNotFoundError is returned by FindLanguage when no directory holds
// a grammar for the language.
type GrammarNotFoundError struct {
	Name string   // the language name
	Dirs []string // the directories searched
}

func (e *GrammarNotFoundError) Error() string {
	return fmt.Sprintf("treesitter: no grammar for %s in %s; set %s or use LoadLanguage",
		e.Name, strings.Join(e.Dirs, string(os.PathListSeparator)), GrammarPathEnv)
}

// FindGrammar returns the path of the shared library for the named language
// in the first directory of [GrammarDirs] that has one. It accepts the file
// names used by the tree-sitter CLI and Neovim (<name>.so or .dylib), by
// Emacs and Linux distributions (libtree-sitter-<name>), and their
// hyphenated variants.
func FindGrammar(name string) (string, error) {
	if name == "" {
		return "", errors.New("treesitter: empty language name")
	}
	dirs := GrammarDirs()
	for _, dir := range dirs {
		for _, base := range grammarFileNames(name) {
			path := filepath.Join(dir, base)
			if fi, err := os.Stat(path); err == nil && !fi.IsDir() {
				return path, nil
			}
		}
	}
	return "", &GrammarNotFoundError{Name: name, Dirs: dirs}
}

// grammarFileNames lists candidate file names for a language, most
// specific first.
func grammarFileNames(name string) []string {
	exts := []string{".so", ".dylib"}
	if runtime.GOOS == "darwin" {
		exts = []string{".dylib", ".so"}
	}
	stems := []string{name}
	if alt := strings.ReplaceAll(name, "_", "-"); alt != name {
		stems = append(stems, alt)
	}
	var out []string
	for _, stem := range stems {
		for _, ext := range exts {
			out = append(out, stem+ext, "libtree-sitter-"+stem+ext, "tree-sitter-"+stem+ext)
		}
	}
	return out
}

// FindLanguage locates and loads the grammar for the named language. The
// name is the grammar's own, as in tree_sitter_<name>: "c", "cpp",
// "c_sharp", "typescript". Set TREE_SITTER_GRAMMAR_PATH to search extra
// directories; see [GrammarDirs] for the full list. Loaded languages are
// cached, so repeated calls with the same name return the same *Language.
func FindLanguage(name string) (*Language, error) {
	if l, ok := foundLanguages.Load(name); ok {
		return l.(*Language), nil
	}
	path, err := FindGrammar(name)
	if err != nil {
		return nil, err
	}
	lang, err := LoadLanguage(path, name)
	if err != nil {
		return nil, err
	}
	l, _ := foundLanguages.LoadOrStore(name, lang)
	return l.(*Language), nil
}
