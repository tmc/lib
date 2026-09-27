package cudacache

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
)

// A mapping is one file named in /proc/self/maps (see proc(5)).
type mapping struct {
	path         string
	major, minor uint64
	inode        uint64
	deleted      bool
}

// mappedFiles returns the distinct files in a /proc/self/maps table whose
// path satisfies match. A shared library is mapped once per segment, so one
// file appears on several lines; those collapse to one mapping.
// Malformed lines and anonymous mappings are skipped.
func mappedFiles(maps []byte, match func(path string) bool) []mapping {
	var out []mapping
	seen := make(map[mapping]bool)
	for _, line := range bytes.Split(maps, []byte("\n")) {
		m, ok := parseMapsLine(string(line))
		if !ok || !match(m.path) || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	return out
}

// parseMapsLine parses "address perms offset dev inode pathname". The
// pathname is the rest of the line after the fifth field and may contain
// spaces; the kernel appends " (deleted)" to a file that was unlinked.
func parseMapsLine(line string) (mapping, bool) {
	rest := line
	var fields [5]string
	for i := range fields {
		rest = strings.TrimLeft(rest, " ")
		j := strings.IndexByte(rest, ' ')
		if j < 0 {
			return mapping{}, false
		}
		fields[i], rest = rest[:j], rest[j:]
	}
	path := strings.TrimLeft(rest, " ")
	if !strings.HasPrefix(path, "/") {
		return mapping{}, false
	}
	var m mapping
	if p, ok := strings.CutSuffix(path, " (deleted)"); ok {
		path, m.deleted = p, true
	}
	m.path = path
	major, minor, ok := strings.Cut(fields[3], ":")
	if !ok {
		return mapping{}, false
	}
	var err error
	if m.major, err = strconv.ParseUint(major, 16, 32); err != nil {
		return mapping{}, false
	}
	if m.minor, err = strconv.ParseUint(minor, 16, 32); err != nil {
		return mapping{}, false
	}
	if m.inode, err = strconv.ParseUint(fields[4], 10, 64); err != nil || m.inode == 0 {
		return mapping{}, false
	}
	return m, true
}

// libraryMatches returns the distinct files in maps that can be the library
// dlopen found for soname. A file named soname, or soname with a longer
// version ("libnvrtc.so.13" for "libnvrtc.so.13.0.88"), is preferred. Only
// if there is none does any version of the same library match, because the
// soname can link to a version that does not extend it ("libcuda.so.1" to
// "libcuda.so.580.95.05"). "libnvrtc-builtins.so.13.0.88" never matches
// "libnvrtc.so.13".
func libraryMatches(maps []byte, soname string) []mapping {
	if ms := mappedFiles(maps, sonameMatch(soname, soname)); len(ms) > 0 {
		return ms
	}
	stem := soname
	if i := strings.Index(soname, ".so."); i >= 0 {
		stem = soname[:i+len(".so")]
	}
	return mappedFiles(maps, sonameMatch(soname, stem))
}

// sonameMatch reports whether a path's base name is soname or starts with
// prefix followed by a dot.
func sonameMatch(soname, prefix string) func(string) bool {
	return func(path string) bool {
		base := filepath.Base(path)
		return base == soname || strings.HasPrefix(base, prefix+".")
	}
}

// uniqueMapping returns the one mapping in ms, or an error naming name.
func uniqueMapping(name string, ms []mapping) (mapping, error) {
	switch len(ms) {
	case 0:
		return mapping{}, fmt.Errorf("loaded library %s: not mapped", name)
	case 1:
		return ms[0], nil
	}
	paths := make([]string, len(ms))
	for i, m := range ms {
		paths[i] = m.path
	}
	return mapping{}, fmt.Errorf("loaded library %s: ambiguous: %s", name, strings.Join(paths, ", "))
}
