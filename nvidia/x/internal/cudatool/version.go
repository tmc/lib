package cudatool

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
)

// Version is a CUDA toolkit version. The zero value is not a version.
type Version struct {
	Major, Minor int
}

// V returns the version major.minor.
func V(major, minor int) Version { return Version{major, minor} }

func (v Version) String() string { return fmt.Sprintf("%d.%d", v.Major, v.Minor) }

// Valid reports whether v names a version.
func (v Version) Valid() bool { return v.Major > 0 }

// Compare orders versions, returning -1, 0 or 1 as v is less than, equal to or
// greater than w.
func (v Version) Compare(w Version) int {
	switch {
	case v.Major != w.Major:
		if v.Major < w.Major {
			return -1
		}
		return 1
	case v.Minor != w.Minor:
		if v.Minor < w.Minor {
			return -1
		}
		return 1
	}
	return 0
}

// releaseLine matches the release the CUDA binary utilities report, as in
// "Cuda compilation tools, release 13.4, V13.4.59".
var releaseLine = regexp.MustCompile(`release (\d+)\.(\d+)`)

// ParseVersion returns the toolkit version named in a tool's --version output.
//
// compute-sanitizer is the exception: it reports its own product version, as in
// "Version 2026.3.0.0 (build 38637409)", which names no toolkit. ParseVersion
// returns false for it, leaving the caller to resolve the version another way.
func ParseVersion(out []byte) (Version, bool) {
	m := releaseLine.FindSubmatch(out)
	if m == nil {
		return Version{}, false
	}
	major, err := strconv.Atoi(string(m[1]))
	if err != nil {
		return Version{}, false
	}
	minor, err := strconv.Atoi(string(m[2]))
	if err != nil {
		return Version{}, false
	}
	return Version{major, minor}, true
}

// prefixVersion matches the version in a toolkit install prefix, as in
// /usr/local/cuda-13.4/bin/compute-sanitizer.
var prefixVersion = regexp.MustCompile(`cuda-(\d+)\.(\d+)(/|$)`)

// VersionFromPath returns the toolkit version named by the install prefix an
// executable sits in. It is how compute-sanitizer's toolkit version is
// recovered, since the tool itself does not report one.
//
// Symbolic links are resolved first: a multi-version install is usually reached
// through /usr/local/cuda, whose own name carries no version while its target's
// does.
func VersionFromPath(path string) (Version, bool) {
	m := prefixVersion.FindStringSubmatch(path)
	if m == nil {
		if resolved, err := filepath.EvalSymlinks(path); err == nil {
			m = prefixVersion.FindStringSubmatch(resolved)
		}
	}
	if m == nil {
		return Version{}, false
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	return Version{major, minor}, true
}
