package grammars

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/tmc/lib/treesitter"
)

//go:generate go run mkmanifest.go

// A Grammar names a tree-sitter grammar and pins the source archive it is
// built from. The zero value is not usable.
type Grammar struct {
	Name    string // grammar name, as in tree_sitter_<name>
	Version string // the upstream release the pin refers to
	URL     string // gzipped tar archive of the grammar's source
	SHA256  string // hex SHA-256 of the archive
	Subdir  string // directory within the archive holding src, "" for the root
}

// String returns the grammar's name and version.
func (g Grammar) String() string { return g.Name + " " + g.Version }

// All returns the pinned grammars, ordered by name.
func All() []Grammar { return slices.Clone(pinned) }

// Lookup returns the pinned grammar with the given name.
func Lookup(name string) (Grammar, bool) {
	i := slices.IndexFunc(pinned, func(g Grammar) bool { return g.Name == name })
	if i < 0 {
		return Grammar{}, false
	}
	return pinned[i], true
}

// Dir returns the directory Install writes to, which is TREE_SITTER_LIBDIR
// if it is set and the tree-sitter cache directory otherwise. It is one of
// the directories [treesitter.GrammarDirs] searches.
func Dir() (string, error) {
	if d := os.Getenv("TREE_SITTER_LIBDIR"); d != "" {
		return d, nil
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("grammars: %w", err)
	}
	return filepath.Join(cache, "tree-sitter", "lib"), nil
}

// Install builds the named grammar into [Dir] and returns the path of the
// shared library. It does nothing if that directory already holds the
// grammar; remove the file to force a rebuild.
//
// Only names reported by [All] can be installed this way. Installing
// downloads the pinned archive, so it needs network access, and compiles
// it, so it needs a C compiler.
func Install(ctx context.Context, name string) (string, error) {
	g, ok := Lookup(name)
	if !ok {
		return "", fmt.Errorf("grammars: no pinned grammar named %s", name)
	}
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	return g.Build(ctx, dir)
}

// Build downloads the grammar's archive, checks it against g.SHA256,
// compiles it, and writes the shared library to dir, which it creates if
// needed. It returns the library's path.
//
// Build reuses a library that is already there. It verifies what it builds
// by loading it, so a returned path names a grammar this package can use.
func (g Grammar) Build(ctx context.Context, dir string) (string, error) {
	if g.Name == "" || g.URL == "" || g.SHA256 == "" {
		return "", errors.New("grammars: grammar needs a name, a URL, and a checksum")
	}
	out := filepath.Join(dir, g.Name+libExt)
	if _, err := os.Stat(out); err == nil {
		return out, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("grammars: %w", err)
	}

	src, err := os.MkdirTemp("", "tree-sitter-"+g.Name)
	if err != nil {
		return "", fmt.Errorf("grammars: %w", err)
	}
	defer os.RemoveAll(src)

	archive, err := g.fetch(ctx)
	if err != nil {
		return "", err
	}
	if err := extract(archive, src); err != nil {
		return "", fmt.Errorf("grammars: %s: %w", g.Name, err)
	}
	if err := g.compile(ctx, filepath.Join(src, filepath.FromSlash(g.Subdir), "src"), out); err != nil {
		return "", err
	}
	if _, err := treesitter.LoadLanguage(out, g.Name); err != nil {
		os.Remove(out)
		return "", fmt.Errorf("grammars: %s built but does not load: %w", g.Name, err)
	}
	return out, nil
}

// libExt is the file name extension for a shared library, chosen to match
// the names treesitter.FindGrammar looks for.
var libExt = func() string {
	if runtime.GOOS == "darwin" {
		return ".dylib"
	}
	return ".so"
}()

// maxArchive bounds a download, so that a wrong or hostile URL cannot fill
// the disk before the checksum rejects it.
const maxArchive = 64 << 20

// fetch downloads the archive and returns it once its digest matches.
func (g Grammar) fetch(ctx context.Context) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, g.URL, nil)
	if err != nil {
		return nil, fmt.Errorf("grammars: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("grammars: fetch %s: %w", g.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("grammars: fetch %s: %s", g.URL, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxArchive+1))
	if err != nil {
		return nil, fmt.Errorf("grammars: fetch %s: %w", g.URL, err)
	}
	if len(body) > maxArchive {
		return nil, fmt.Errorf("grammars: fetch %s: archive is larger than %d bytes", g.URL, maxArchive)
	}
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != g.SHA256 {
		return nil, fmt.Errorf("grammars: %s: checksum mismatch for %s:\n\thave %s\n\twant %s",
			g.Name, g.URL, got, g.SHA256)
	}
	return body, nil
}

// extract unpacks a gzipped tar into dir. It takes only regular files and
// directories, and rejects any name that would escape dir.
func extract(archive []byte, dir string) error {
	zr, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		return err
	}
	defer zr.Close()
	tr := tar.NewReader(zr)
	var total int64
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		name := path.Clean(strings.TrimPrefix(h.Name, "./"))
		if path.IsAbs(name) || name == ".." || strings.HasPrefix(name, "../") {
			return fmt.Errorf("archive entry %q is outside the archive", h.Name)
		}
		// GitHub archives wrap everything in one directory named for the
		// repository and tag, which is not part of the layout the grammar
		// declares, so drop it.
		i := strings.Index(name, "/")
		if i < 0 {
			continue
		}
		if name = name[i+1:]; name == "" {
			continue
		}
		target := filepath.Join(dir, filepath.FromSlash(name))
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			total += h.Size
			if total > maxArchive {
				return fmt.Errorf("archive expands to more than %d bytes", maxArchive)
			}
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			f, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
			if err != nil {
				return err
			}
			_, err = io.Copy(f, io.LimitReader(tr, maxArchive))
			if cerr := f.Close(); err == nil {
				err = cerr
			}
			if err != nil {
				return err
			}
		}
	}
}

// compile builds the grammar's sources in srcDir into the shared library
// out, writing to a temporary file first so that a failed build leaves no
// half-written library behind.
func (g Grammar) compile(ctx context.Context, srcDir, out string) error {
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		return fmt.Errorf("grammars: %s needs a C compiler: %w", g.Name, err)
	}
	sources := []string{filepath.Join(srcDir, "parser.c")}
	if _, err := os.Stat(sources[0]); err != nil {
		return fmt.Errorf("grammars: %s: %w", g.Name, err)
	}
	// A grammar with tokens its rules cannot express carries an external
	// scanner beside the generated parser.
	if scanner := filepath.Join(srcDir, "scanner.c"); fileExists(scanner) {
		sources = append(sources, scanner)
	}

	tmp, err := os.CreateTemp(filepath.Dir(out), "."+g.Name+"-*"+libExt)
	if err != nil {
		return fmt.Errorf("grammars: %w", err)
	}
	tmp.Close()
	defer os.Remove(tmp.Name())

	args := []string{"-shared", "-fPIC", "-O2", "-I", srcDir, "-o", tmp.Name()}
	args = append(args, sources...)
	cmd := exec.CommandContext(ctx, cc, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("grammars: %s: %s %s: %w\n%s", g.Name, cc, strings.Join(args, " "), err, out)
	}
	if err := os.Rename(tmp.Name(), out); err != nil {
		return fmt.Errorf("grammars: %w", err)
	}
	return nil
}

func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}
