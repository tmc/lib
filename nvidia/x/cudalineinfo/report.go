package cudalineinfo

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
)

// WriteTable writes one row per site: the site, its source line as written on
// disk, and how many generated CUDA lines, PTX locations, and SASS
// instructions name it, followed by the remainder.
func WriteTable(w io.Writer, a *Attribution) error {
	src := newSourceReader()
	names := SiteNames(a.Sites())
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: %d of %d SASS instructions name a program line\n", a.Unit.Name, a.Attributed, a.Instructions)
	tw := tabwriter.NewWriter(bw, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  SITE\tSOURCE\tCUDA\tPTX\tSASS")
	for _, r := range a.Rows() {
		fmt.Fprintf(tw, "  %s\t%s\t%d\t%d\t%d\n", names[r.Site], src.line(r.Site), len(r.CUDA), r.PTX, len(r.SASS))
	}
	tw.Flush()
	fmt.Fprintf(bw, "  remainder: %s\n", a.Remainder)
	return bw.Flush()
}

// WriteListing writes the attributed instructions, grouped by site.
func WriteListing(w io.Writer, a *Attribution) error {
	src := newSourceReader()
	names := SiteNames(a.Sites())
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: SASS by program line\n", a.Unit.Name)
	for _, s := range a.Sites() {
		insts := a.SASS[s]
		if len(insts) == 0 {
			continue
		}
		fmt.Fprintf(bw, "  %s  %s\n", names[s], src.line(s))
		for _, in := range insts {
			op := in.Opcode
			if in.Predicate != "" {
				op = in.Predicate + " " + op
			}
			fmt.Fprintf(bw, "      /*%04x*/  %-24s %s\n", in.Address, op, in.Operands)
		}
	}
	return bw.Flush()
}

// A sourceReader returns lines of source files, reading each file once.
type sourceReader struct {
	files map[string][]string
}

func newSourceReader() *sourceReader {
	return &sourceReader{files: map[string][]string{}}
}

func (r *sourceReader) line(s Site) string {
	lines, ok := r.files[s.File]
	if !ok {
		lines = readLines(s.File)
		r.files[s.File] = lines
	}
	if lines == nil {
		return ""
	}
	if s.Line < 1 || s.Line > len(lines) {
		return "<past end of file>"
	}
	return strings.TrimSpace(lines[s.Line-1])
}

// readLines returns the lines of a file, or nil when it cannot be read: a
// site in a file that is not on this machine is still a site, and the table
// shows it with an empty source column rather than failing.
func readLines(path string) []string {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	return lines
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
