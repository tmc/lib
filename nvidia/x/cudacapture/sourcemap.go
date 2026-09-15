package cudacapture

import (
	"fmt"
	"path"
	"strings"
)

// Line directives nvdisasm emits when a CUBIN carries line information. The
// first names the source file a machine instruction came from; the second
// appears on the same line once per inlining level.
const (
	fileDirective   = `//## File "`
	inlinedAtMarker = ` inlined at "`
)

// A SourceMap sends a compile-time source path, as the compiler recorded it in
// the CUBIN, to the name a capture bundles that file under. A path the map does
// not name is left alone: the capture has nothing to say about it, and claiming
// otherwise would point a reader at the wrong file.
type SourceMap map[string]string

// RewriteSASSSources returns sass with every source path in an nvdisasm line
// directive replaced by the artifact name m gives it, including the paths in
// inlining annotations. Paths m does not name are retained byte for byte.
//
// The compiler records paths from the build host, which resolve to nothing on a
// reader's machine, so a capture that bundles its sources must rewrite the
// directives to the names it bundles them under. Only the paths the caller can
// account for are rewritten: nvdisasm also names files the capture never saw
// and the pseudo-file ".nv_debug_ptx_txt", which holds the embedded PTX text
// rather than a source file, and all of those survive unchanged.
func RewriteSASSSources(sass []byte, m SourceMap) []byte {
	out := make([]byte, 0, len(sass))
	appendMapped := func(p string) {
		if name, ok := m[p]; ok {
			out = append(out, name...)
			return
		}
		out = append(out, p...)
	}
	for _, line := range strings.SplitAfter(string(sass), "\n") {
		i := strings.Index(line, fileDirective)
		if i < 0 {
			out = append(out, line...)
			continue
		}
		rest := line[i+len(fileDirective):]
		j := strings.IndexByte(rest, '"')
		if j < 0 {
			out = append(out, line...)
			continue
		}
		out = append(out, line[:i+len(fileDirective)]...)
		appendMapped(rest[:j])

		remaining := rest[j:]
		for {
			k := strings.Index(remaining, inlinedAtMarker)
			if k < 0 {
				out = append(out, remaining...)
				break
			}
			out = append(out, remaining[:k+len(inlinedAtMarker)]...)
			inlineRest := remaining[k+len(inlinedAtMarker):]
			end := strings.IndexByte(inlineRest, '"')
			if end < 0 {
				out = append(out, inlineRest...)
				break
			}
			appendMapped(inlineRest[:end])
			remaining = inlineRest[end:]
		}
	}
	return out
}

// SASSSourcePaths returns the distinct source paths nvdisasm recorded in sass,
// in order of first appearance. Inlining annotations are included, since an
// inlined frame names a file a reader resolves like any other.
func SASSSourcePaths(sass []byte) []string {
	var paths []string
	seen := make(map[string]bool)
	add := func(p string) {
		if !seen[p] {
			seen[p] = true
			paths = append(paths, p)
		}
	}
	for _, line := range strings.SplitAfter(string(sass), "\n") {
		i := strings.Index(line, fileDirective)
		if i < 0 {
			continue
		}
		rest := line[i+len(fileDirective):]
		j := strings.IndexByte(rest, '"')
		if j < 0 {
			continue
		}
		add(rest[:j])
		remaining := rest[j:]
		for {
			k := strings.Index(remaining, inlinedAtMarker)
			if k < 0 {
				break
			}
			inlineRest := remaining[k+len(inlinedAtMarker):]
			end := strings.IndexByte(inlineRest, '"')
			if end < 0 {
				break
			}
			add(inlineRest[:end])
			remaining = inlineRest[end:]
		}
	}
	return paths
}

// SourceMapByBase returns a SourceMap sending each source path recorded in sass
// to the artifact of the same base name. It is the fallback for a build whose
// recorded paths are not known until the compiler has run, such as a compile
// through a temporary directory.
//
// It is deliberately not what RewriteSASSSources does by default. Base names do
// not identify a source file: two directories can each hold a kernels.cu, and a
// rewrite that merged them would attribute one file's instructions to the
// other. SourceMapByBase reports an error rather than choosing between two
// recorded paths with the same base name, so a caller that cannot tell them
// apart gets no map instead of a wrong one.
func SourceMapByBase(sass []byte, artifacts ...string) (SourceMap, error) {
	want := make(map[string]string, len(artifacts))
	for _, a := range artifacts {
		want[path.Base(a)] = a
	}
	m := make(SourceMap)
	claimed := make(map[string]string)
	for _, p := range SASSSourcePaths(sass) {
		base := path.Base(p)
		artifact, ok := want[base]
		if !ok {
			continue
		}
		if other, dup := claimed[base]; dup {
			return nil, fmt.Errorf("cudacapture: base name %q names two recorded sources %q and %q", base, other, p)
		}
		claimed[base] = p
		m[p] = artifact
	}
	return m, nil
}
