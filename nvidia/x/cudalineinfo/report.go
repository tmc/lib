package cudalineinfo

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
)

// WriteTable writes one row per site: the site, its source line as written on
// disk, and how many generated CUDA lines, PTX locations, and SASS
// instructions name it, followed by the remainder.
func WriteTable(w io.Writer, a *Attribution) error {
	src := newSourceReader(a.Unit)
	names := SiteNames(a.Sites())
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: %d of %d SASS instructions name a program line\n", a.Unit.Name, a.Attributed, a.Instructions)
	writeUnattributed(bw, a)
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
	src := newSourceReader(a.Unit)
	names := SiteNames(a.Sites())
	bw := bufio.NewWriter(w)
	fmt.Fprintf(bw, "%s: SASS by program line\n", a.Unit.Name)
	writeUnattributed(bw, a)
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

// writeUnattributed says that a report is empty because the join failed,
// which is otherwise indistinguishable from a program with no hot lines.
func writeUnattributed(w io.Writer, a *Attribution) {
	if !a.Unattributed() {
		return
	}
	fmt.Fprintf(w, "  nothing attributed: none of the %d instructions names a program line; the unit is %s\n", a.Instructions, unitDescription(a.Unit))
}

// unitDescription says what the unit brought to the join, which is what
// decides whether anything could have been attributed.
func unitDescription(u Unit) string {
	switch {
	case u.Source == "" && len(u.LineMap) == 0:
		return "bare, holding only the name " + strconv.Quote(u.Name)
	case len(u.LineMap) == 0:
		return "hand-written " + u.GeneratedName()
	}
	return "generated " + u.GeneratedName()
}

// A sourceReader returns lines of source files, reading each file once. The
// unit's own source comes from the unit rather than from disk: with NVRTC it
// is often embedded in the program, and there is no file to open.
type sourceReader struct {
	unit  Unit
	files map[string][]string
}

func newSourceReader(u Unit) *sourceReader {
	r := &sourceReader{unit: u, files: map[string][]string{}}
	if u.Source != "" {
		r.files[u.GeneratedName()] = strings.Split(u.Source, "\n")
	}
	return r
}

func (r *sourceReader) line(s Site) string {
	file := s.File
	if r.unit.Source != "" && filepath.Base(file) == r.unit.GeneratedName() {
		file = r.unit.GeneratedName()
	}
	lines, ok := r.files[file]
	if !ok {
		lines = readLines(file)
		r.files[file] = lines
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
