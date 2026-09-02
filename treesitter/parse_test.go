package treesitter

import (
	"errors"
	"strings"
	"testing"
)

// recordingReaderAt records the offsets tree-sitter asks for and can fail
// after a set number of reads.
type recordingReaderAt struct {
	r       *strings.Reader
	offs    []int64
	failAt  int // read number to fail on, counting from 1; 0 never fails
	failErr error
}

func (rr *recordingReaderAt) ReadAt(b []byte, off int64) (int, error) {
	rr.offs = append(rr.offs, off)
	n, err := rr.r.ReadAt(b, off)
	if rr.failAt > 0 && len(rr.offs) >= rr.failAt {
		// Report a short read with a real error, the shape a failing
		// file has. The bytes read so far must not be mistaken for a
		// successful chunk: doing so parses truncated input silently.
		return min(n, 64), rr.failErr
	}
	return n, err
}

// longSource returns C source with one declaration long enough to span
// several read windows, so that a parse must revisit earlier offsets.
func longSource() string {
	var sb strings.Builder
	sb.WriteString("int f(")
	for sb.Len() < 2*readAtChunk {
		sb.WriteString("int aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, ")
	}
	sb.WriteString("int z) { return 0; }\n")
	return sb.String()
}

// TestParseReaderAtBackwardReads is the reason ParseReaderAt takes an
// io.ReaderAt: the lexer backs up to the start of the current line to
// compute a column, so it reads offsets it has already passed. A
// forward-only io.Reader cannot serve that.
func TestParseReaderAtBackwardReads(t *testing.T) {
	p, _ := testParser(t)
	src := longSource()
	rr := &recordingReaderAt{r: strings.NewReader(src)}
	tree, err := p.ParseReaderAt(rr, int64(len(src)), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if tree.HasError() {
		t.Error("parse error")
	}
	if got, want := tree.RootNode().EndByte(), len(src); got != want {
		t.Errorf("end byte = %d, want %d", got, want)
	}

	back := false
	for i, off := range rr.offs {
		if i > 0 && off < rr.offs[i-1] {
			back = true
		}
	}
	if !back {
		t.Errorf("no backward read in offsets %v; ParseReaderAt may no longer need random access", rr.offs)
	}
}

func TestParseReaderAtErrors(t *testing.T) {
	p, _ := testParser(t)
	src := longSource()

	// A read error stops the parse and is reported.
	want := errors.New("disk on fire")
	rr := &recordingReaderAt{r: strings.NewReader(src), failAt: 2, failErr: want}
	tree, err := p.ParseReaderAt(rr, int64(len(src)), nil)
	if !errors.Is(err, want) {
		t.Errorf("err = %v, want %v", err, want)
	}
	if tree != nil {
		tree.Close()
		t.Error("ParseReaderAt returned a tree after a read error")
	}

	// A truncated read that reports io.EOF ends the input without error.
	short := &recordingReaderAt{r: strings.NewReader("int x;")}
	tree, err = p.ParseReaderAt(short, 1<<20, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()
	if got, want := tree.RootNode().EndByte(), len("int x;"); got != want {
		t.Errorf("end byte = %d, want %d", got, want)
	}

	// tree-sitter addresses input with a uint32.
	if _, err := p.ParseReaderAt(strings.NewReader(""), 1<<32, nil); err == nil {
		t.Error("ParseReaderAt accepted a size past the 4 GiB limit")
	}
	if _, err := p.ParseReaderAt(strings.NewReader(""), int64(^uint32(0)), nil); err != nil {
		t.Errorf("ParseReaderAt rejected the largest allowed size: %v", err)
	}
}

func TestTreeSource(t *testing.T) {
	p, _ := testParser(t)
	src := []byte("int x;")
	tree, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Close()

	// A copy shares the original's source, and outlives it.
	cp := tree.Copy()
	defer cp.Close()
	if got := string(cp.Source()); got != string(src) {
		t.Errorf("copy Source() = %q, want %q", got, src)
	}
	if got := string(cp.RootNode().NamedChild(0).Text()); got != "int x;" {
		t.Errorf("copy Text() = %q", got)
	}

	// Edit records a change for the next parse but does not restate the
	// source, so Text still reads the text the tree was parsed from.
	tree.Edit(InputEdit{
		StartByte: 4, OldEndByte: 5, NewEndByte: 9,
		StartPoint:  Point{Row: 0, Column: 4},
		OldEndPoint: Point{Row: 0, Column: 5},
		NewEndPoint: Point{Row: 0, Column: 9},
	})
	if got := string(tree.Source()); got != string(src) {
		t.Errorf("Source() after Edit = %q, want %q", got, src)
	}

	// Edit moved the root past the end of the retained source, so Text
	// reports no text rather than slicing out of range.
	if got := tree.RootNode().Text(); got != nil {
		t.Errorf("Text() after Edit = %q, want nil", got)
	}

	// The reparse carries the new source.
	next, err := p.Parse([]byte("int count;"), tree)
	if err != nil {
		t.Fatal(err)
	}
	defer next.Close()
	if got := string(next.RootNode().Text()); got != "int count;" {
		t.Errorf("reparsed Text() = %q", got)
	}
}
