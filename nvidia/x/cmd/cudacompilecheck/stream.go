package main

import (
	"archive/tar"
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
)

// remote returns the path on the compiling host of the local file or
// directory p. When sources are streamed, the compiling host holds a copy
// of each under c.root at its local absolute path; otherwise it sees p
// itself.
func (c *config) remote(p string) string {
	if c.root == "" {
		return p
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return c.root + filepath.ToSlash(abs)
}

// send copies files to the compiling host under c.root, as a tar archive
// on the -exec command's standard input, so that no filesystem need be
// shared with it.
func (c *config) send(ctx context.Context, files []string) error {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return err
		}
		abs, err := filepath.Abs(f)
		if err != nil {
			return err
		}
		hdr := &tar.Header{Name: filepath.ToSlash(abs)[1:], Mode: 0o644, Size: int64(len(data))}
		if err := tw.WriteHeader(hdr); err != nil {
			return err
		}
		if _, err := tw.Write(data); err != nil {
			return err
		}
	}
	if err := tw.Close(); err != nil {
		return err
	}
	if out, err := c.run(ctx, "mkdir", "-p", c.root); err != nil {
		return fmt.Errorf("make source directory: %v\n%s", err, out)
	}
	if out, err := c.runOn(ctx, false, buf.Bytes(), []string{"tar", "-xf", "-", "-C", c.root}); err != nil {
		return fmt.Errorf("send sources: %v\n%s", err, out)
	}
	return nil
}

var includeRE = regexp.MustCompile(`(?m)^[ \t]*#[ \t]*include[ \t]*([<"])([^>"\n]+)[>"]`)

// includes returns src and every local file it includes, directly or
// not: a quoted name resolved against the including file's directory and
// then dirs, an angled name against dirs. Includes inside #if are
// followed too. A name found in neither place is on the compiling host,
// such as a toolkit header.
func includes(src string, dirs []string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	var visit func(file string) error
	visit = func(file string) error {
		if seen[file] {
			return nil
		}
		seen[file] = true
		out = append(out, file)
		text, err := os.ReadFile(file)
		if err != nil {
			return err
		}
		for _, m := range includeRE.FindAllSubmatch(text, -1) {
			name := string(m[2])
			var try []string
			switch {
			case filepath.IsAbs(name):
				try = []string{name}
			case string(m[1]) == `"`:
				try = append([]string{filepath.Dir(file)}, dirs...)
			default:
				try = dirs
			}
			for _, d := range try {
				p := filepath.Clean(filepath.Join(d, name))
				if filepath.IsAbs(name) {
					p = name
				}
				if st, err := os.Stat(p); err == nil && st.Mode().IsRegular() {
					if err := visit(p); err != nil {
						return err
					}
					break
				}
			}
		}
		return nil
	}
	abs, err := filepath.Abs(src)
	if err != nil {
		return nil, err
	}
	if err := visit(abs); err != nil {
		return nil, err
	}
	return out, nil
}

// files returns the regular files under dir.
func files(dir string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() {
			out = append(out, p)
		}
		return nil
	})
	slices.Sort(out)
	return out, err
}
