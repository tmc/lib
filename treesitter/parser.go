package treesitter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// ErrNoLanguage is returned by Parse when the parser has no language.
var ErrNoLanguage = errors.New("treesitter: parser has no language")

// ErrParseFailed is returned when tree-sitter produces no tree for reasons
// other than a missing language or a cancelled context.
var ErrParseFailed = errors.New("treesitter: parse failed")

// Parser turns source text into a [Tree]. A Parser is not safe for
// concurrent use.
type Parser struct {
	p    ts.Parser
	lang *Language

	// logID keys this parser in the callback registry while a logger is
	// installed.
	logID  uintptr
	logger func(LogType, string)

	// dot records that PrintDotGraphs handed the parser a descriptor that
	// Close must release; ts_parser_delete does not close it.
	dot bool
}

// NewParser returns a parser with no language set.
func NewParser() (*Parser, error) {
	if err := load(); err != nil {
		return nil, err
	}
	p := ts.ParserNew()
	if p == 0 {
		return nil, errors.New("treesitter: ts_parser_new returned NULL")
	}
	return &Parser{p: ts.Parser(p)}, nil
}

// Close frees the parser. Trees it produced remain valid.
func (p *Parser) Close() {
	if p.p == 0 {
		return
	}
	if p.logID != 0 {
		callbacks.Delete(p.logID)
	}
	p.StopPrintingDotGraphs()
	ts.ParserDelete(p.p)
	p.p = 0
}

// SetLanguage selects the grammar to parse with. It fails when the
// grammar's ABI version is incompatible with the library.
func (p *Parser) SetLanguage(lang *Language) error {
	if lang == nil {
		return errors.New("treesitter: SetLanguage: nil language")
	}
	if !ts.ParserSetLanguage(p.p, lang.l) {
		return errors.New("treesitter: SetLanguage: incompatible language ABI version")
	}
	p.lang = lang
	return nil
}

// Language returns the parser's language, or nil if none is set.
func (p *Parser) Language() *Language {
	return p.lang
}

// Parse parses UTF-8 source. old, if non-nil, is a tree from a previous
// parse of the same document, already updated with [Tree.Edit]; passing it
// makes the parse incremental. The source is read only during the call.
func (p *Parser) Parse(src []byte, old *Tree) (*Tree, error) {
	return p.ParseEncoding(src, UTF8, old)
}

// ParseEncoding is like Parse for UTF-8 or UTF-16 source.
func (p *Parser) ParseEncoding(src []byte, enc InputEncoding, old *Tree) (*Tree, error) {
	if p.lang == nil {
		return nil, ErrNoLanguage
	}
	t := ts.ParserParseStringEncoding(p.p, oldTree(old), bytesPtr(src), uint32(len(src)), ts.InputEncoding(enc))
	if t == 0 {
		return nil, ErrParseFailed
	}
	return &Tree{t: ts.Tree(t), src: src}, nil
}

// ParseContext is like Parse but stops when ctx is done, returning ctx's
// error. Cancellation is checked as parsing progresses, so a very small
// input may finish before the context is observed.
func (p *Parser) ParseContext(ctx context.Context, src []byte, old *Tree) (*Tree, error) {
	if p.lang == nil {
		return nil, ErrNoLanguage
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	in := &inputState{src: src, ctx: ctx}
	id := register(in)
	defer callbacks.Delete(id)
	input := ts.Input{
		Payload:  id,
		Read:     readCallback(),
		Encoding: ts.InputEncodingUTF8,
	}
	opts := ts.ParseOptions{Payload: id, ProgressCallback: progressCallback()}
	t := ts.ParserParseWithOptions(p.p, oldTree(old), input, opts)
	if t == 0 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return nil, ErrParseFailed
	}
	return &Tree{t: ts.Tree(t), src: src}, nil
}

// ParseInput parses text supplied by read, which returns the text starting
// at byteIndex, or nil at the end of the document. pos is the same position
// as a point. read may return the text in chunks of any size; returned
// slices are copied before tree-sitter sees them.
//
// tree-sitter reads by byte offset and revisits earlier offsets, so read
// must serve any offset it has already been asked for, not only advancing
// ones. The resulting tree has no source, so [Node.Text] returns nil for
// its nodes.
func (p *Parser) ParseInput(read func(byteIndex int, pos Point) []byte, enc InputEncoding, old *Tree) (*Tree, error) {
	if p.lang == nil {
		return nil, ErrNoLanguage
	}
	in := &inputState{read: read}
	id := register(in)
	defer callbacks.Delete(id)
	input := ts.Input{
		Payload:  id,
		Read:     readCallback(),
		Encoding: ts.InputEncoding(enc),
	}
	t := ts.ParserParse(p.p, oldTree(old), input)
	if t == 0 {
		return nil, ErrParseFailed
	}
	return &Tree{t: ts.Tree(t)}, nil
}

// IncludedRangesError is returned by SetIncludedRanges when a range is
// inverted or begins before the previous range ends. Index is the position
// of the offending range in the slice passed in.
type IncludedRangesError struct {
	Index int
}

func (e *IncludedRangesError) Error() string {
	return fmt.Sprintf("treesitter: included range %d is inverted or overlaps its predecessor", e.Index)
}

// SetIncludedRanges restricts parsing to the given byte ranges of the
// document, for example the code blocks of a template. Ranges must be
// ordered and non-overlapping; a violation is reported as an
// *IncludedRangesError. An empty slice parses the whole document.
func (p *Parser) SetIncludedRanges(ranges []Range) error {
	if len(ranges) == 0 {
		if !ts.ParserSetIncludedRanges(p.p, nil, 0) {
			return errors.New("treesitter: SetIncludedRanges: rejected")
		}
		return nil
	}
	if i := invalidRange(ranges); i >= 0 {
		return &IncludedRangesError{Index: i}
	}
	rs := make([]ts.Range, len(ranges))
	for i, r := range ranges {
		rs[i] = toRange(r)
	}
	if !ts.ParserSetIncludedRanges(p.p, &rs[0], uint32(len(rs))) {
		return errors.New("treesitter: SetIncludedRanges: rejected")
	}
	return nil
}

// invalidRange returns the index of the first range that tree-sitter would
// reject, or -1. tree-sitter requires each range's end to be at or after
// its start and each start to be at or after the previous end.
func invalidRange(ranges []Range) int {
	prevEnd := 0
	for i, r := range ranges {
		if r.EndByte < r.StartByte || r.StartByte < prevEnd {
			return i
		}
		prevEnd = r.EndByte
	}
	return -1
}

// IncludedRanges returns the ranges the parser is restricted to. With no
// restriction it returns one range covering the whole document.
func (p *Parser) IncludedRanges() []Range {
	var n uint32
	ptr := ts.ParserIncludedRanges(p.p, &n)
	return rangeSlice(ptr, n)
}

// Reset discards the state of an interrupted parse so that the next Parse
// starts from the beginning.
func (p *Parser) Reset() {
	ts.ParserReset(p.p)
}

// SetLogger installs a function that receives the parser's log messages,
// or removes it when fn is nil.
func (p *Parser) SetLogger(fn func(kind LogType, msg string)) {
	if p.logID != 0 {
		callbacks.Delete(p.logID)
		p.logID = 0
	}
	p.logger = fn
	if fn == nil {
		ts.ParserSetLogger(p.p, ts.Logger{})
		return
	}
	p.logID = register(p)
	ts.ParserSetLogger(p.p, ts.Logger{Payload: p.logID, Log: logCallback()})
}

// SlogLogger adapts a [slog.Logger] to [Parser.SetLogger]:
//
//	p.SetLogger(treesitter.SlogLogger(logger))
//
// Messages are logged at debug level with a kind attribute of "parse" or
// "lex"; tree-sitter emits one per parse step, so this is verbose.
func SlogLogger(l *slog.Logger) func(LogType, string) {
	if l == nil {
		return nil
	}
	return func(kind LogType, msg string) {
		l.Debug("tree-sitter", "kind", kind.String(), "msg", msg)
	}
}

// PrintDotGraphs makes the parser write a Graphviz graph of its state to
// f after each parse step until StopPrintingDotGraphs or Close. The parser
// writes through its own duplicate of f's descriptor, so f may be closed
// once PrintDotGraphs returns.
func (p *Parser) PrintDotGraphs(f *os.File) error {
	fd, err := syscall.Dup(int(f.Fd()))
	if err != nil {
		return fmt.Errorf("treesitter: PrintDotGraphs: dup: %w", err)
	}
	ts.ParserPrintDotGraphs(p.p, int32(fd))
	p.dot = true
	return nil
}

// StopPrintingDotGraphs stops the output started by PrintDotGraphs and
// closes the parser's duplicate descriptor. Close calls it.
func (p *Parser) StopPrintingDotGraphs() {
	if !p.dot {
		return
	}
	ts.ParserPrintDotGraphs(p.p, -1)
	p.dot = false
}

func oldTree(t *Tree) ts.Tree {
	if t == nil {
		return 0
	}
	return t.t
}

// rangeSlice copies n ranges out of C memory.
func rangeSlice(p uintptr, n uint32) []Range {
	if p == 0 || n == 0 {
		return nil
	}
	src := unsafe.Slice((*ts.Range)(cptr(p)), n)
	out := make([]Range, n)
	for i, r := range src {
		out[i] = fromRange(r)
	}
	return out
}

// Callback plumbing. purego callbacks are created once per signature and
// never released, so each C entry point is a single Go function that
// dispatches on an id passed through the payload pointer.

var (
	callbacks       sync.Map // uintptr -> *Parser, *inputState, or context.Context
	nextID          atomic.Uintptr
	cbOnce          sync.Once
	cbLog           uintptr
	cbRead          uintptr
	cbProgress      uintptr
	cbQueryProgress uintptr
)

func register(v any) uintptr {
	id := nextID.Add(1)
	callbacks.Store(id, v)
	return id
}

func initCallbacks() {
	cbOnce.Do(func() {
		cbLog = purego.NewCallback(onLog)
		cbRead = purego.NewCallback(onRead)
		cbProgress = purego.NewCallback(onProgress)
		cbQueryProgress = purego.NewCallback(onQueryProgress)
	})
}

func logCallback() uintptr      { initCallbacks(); return cbLog }
func readCallback() uintptr     { initCallbacks(); return cbRead }
func progressCallback() uintptr { initCallbacks(); return cbProgress }

func queryProgressCallback() uintptr { initCallbacks(); return cbQueryProgress }

func onLog(payload uintptr, kind ts.LogType, msg uintptr) {
	v, ok := callbacks.Load(payload)
	if !ok {
		return
	}
	if p, ok := v.(*Parser); ok && p.logger != nil {
		p.logger(LogType(kind), goString(msg))
	}
}

// inputState is the Go side of a TSInput or TSParseOptions.
type inputState struct {
	// Exactly one of src and read is set.
	src  []byte
	read func(byteIndex int, pos Point) []byte
	ctx  context.Context

	// buf holds the chunk most recently handed to tree-sitter. Go heap
	// memory does not move, and buf stays reachable through the callback
	// registry until the parse returns, so the pointer stays valid for as
	// long as tree-sitter may read it.
	buf []byte
}

// endOfInput is a byte tree-sitter can point at when a read returns no
// data. A zero count marks the end, but a non-null pointer is friendlier.
var endOfInput byte

func onRead(payload uintptr, byteIndex uint32, pos ts.Point, bytesRead *uint32) uintptr {
	v, ok := callbacks.Load(payload)
	if !ok {
		*bytesRead = 0
		return uintptr(unsafe.Pointer(&endOfInput))
	}
	in := v.(*inputState)
	var chunk []byte
	if in.read != nil {
		chunk = in.read(int(byteIndex), fromPoint(pos))
	} else if int(byteIndex) < len(in.src) {
		chunk = in.src[byteIndex:]
	}
	if len(chunk) == 0 {
		*bytesRead = 0
		return uintptr(unsafe.Pointer(&endOfInput))
	}
	if in.read != nil {
		in.buf = append(in.buf[:0], chunk...)
		chunk = in.buf
	}
	*bytesRead = uint32(len(chunk))
	return uintptr(unsafe.Pointer(unsafe.SliceData(chunk)))
}

func onProgress(state *ts.ParseState) bool {
	v, ok := callbacks.Load(state.Payload)
	if !ok {
		return false
	}
	in := v.(*inputState)
	return in.ctx != nil && in.ctx.Err() != nil
}

func onQueryProgress(state *ts.QueryCursorState) bool {
	v, ok := callbacks.Load(state.Payload)
	if !ok {
		return false
	}
	ctx := v.(context.Context)
	return ctx.Err() != nil
}
