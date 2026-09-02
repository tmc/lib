package treesitter

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"slices"
	"strings"
	"unsafe"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// QueryErrorKind classifies a query compilation error.
type QueryErrorKind int32

// Query error kinds.
const (
	QueryErrorSyntax    QueryErrorKind = QueryErrorKind(ts.QueryErrorSyntax)
	QueryErrorNodeType  QueryErrorKind = QueryErrorKind(ts.QueryErrorNodeType)
	QueryErrorField     QueryErrorKind = QueryErrorKind(ts.QueryErrorField)
	QueryErrorCapture   QueryErrorKind = QueryErrorKind(ts.QueryErrorCapture)
	QueryErrorStructure QueryErrorKind = QueryErrorKind(ts.QueryErrorStructure)
	QueryErrorLanguage  QueryErrorKind = QueryErrorKind(ts.QueryErrorLanguage)
)

// String returns the name of the error kind.
func (k QueryErrorKind) String() string {
	switch k {
	case QueryErrorSyntax:
		return "syntax"
	case QueryErrorNodeType:
		return "node type"
	case QueryErrorField:
		return "field"
	case QueryErrorCapture:
		return "capture"
	case QueryErrorStructure:
		return "structure"
	case QueryErrorLanguage:
		return "language"
	}
	return fmt.Sprintf("QueryErrorKind(%d)", int32(k))
}

// QueryError reports why a query failed to compile and where. Offset is
// the byte offset into the query source; Row and Column locate the same
// byte as a zero-based line and column. Message, when set, says what was
// wrong with a predicate.
type QueryError struct {
	Kind    QueryErrorKind
	Offset  int
	Row     int
	Column  int
	Message string
}

func (e *QueryError) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("treesitter: query %s error at %d:%d (byte %d): %s", e.Kind, e.Row, e.Column, e.Offset, e.Message)
	}
	return fmt.Sprintf("treesitter: query %s error at %d:%d (byte %d)", e.Kind, e.Row, e.Column, e.Offset)
}

// newQueryError locates offset within source.
func newQueryError(kind QueryErrorKind, source string, offset int) *QueryError {
	e := &QueryError{Kind: kind, Offset: offset}
	if offset > len(source) {
		offset = len(source)
	}
	if offset < 0 {
		offset = 0
	}
	before := source[:offset]
	e.Row = strings.Count(before, "\n")
	e.Column = len(before) - (strings.LastIndexByte(before, '\n') + 1)
	return e
}

// Query is a compiled set of tree patterns. A query owns C memory and
// must be closed.
type Query struct {
	q ts.Query

	// names indexes capture names by capture id, so that a Match can
	// report names without calling into C.
	names []string

	// preds and props hold the parsed predicates and properties of each
	// pattern, in pattern order; text holds the compiled form of the
	// predicates Satisfies evaluates.
	preds [][]Predicate
	props [][]Property
	text  [][]textPredicate
}

// NewQuery compiles source, in tree-sitter's query language, for lang.
// A compile failure is reported as a *QueryError.
func NewQuery(lang *Language, source string) (*Query, error) {
	if lang == nil {
		return nil, errors.New("treesitter: NewQuery: nil language")
	}
	var offset uint32
	var kind ts.QueryError
	q := ts.QueryNew(lang.l, stringPtr(source), uint32(len(source)), &offset, &kind)
	if q == 0 {
		return nil, newQueryError(QueryErrorKind(kind), source, int(offset))
	}
	out := &Query{q: ts.Query(q)}
	names := make([]string, out.CaptureCount())
	for i := range names {
		var n uint32
		names[i] = ts.QueryCaptureNameForId(out.q, u32(i), &n)
	}
	out.names = names
	if err := out.parsePredicates(source); err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

// NewQueryFS compiles the query stored at name in fsys. Queries are
// ordinary text, so a query set can be kept in files and embedded with
// go:embed.
func NewQueryFS(lang *Language, fsys fs.FS, name string) (*Query, error) {
	b, err := fs.ReadFile(fsys, name)
	if err != nil {
		return nil, fmt.Errorf("treesitter: read query: %w", err)
	}
	q, err := NewQuery(lang, string(b))
	if err != nil {
		return nil, fmt.Errorf("treesitter: %s: %w", name, err)
	}
	return q, nil
}

// Close frees the query.
func (q *Query) Close() {
	if q.q == 0 {
		return
	}
	ts.QueryDelete(q.q)
	q.q = 0
}

// PatternCount returns the number of patterns.
func (q *Query) PatternCount() int { return int(ts.QueryPatternCount(q.q)) }

// CaptureCount returns the number of distinct capture names.
func (q *Query) CaptureCount() int { return int(ts.QueryCaptureCount(q.q)) }

// StringCount returns the number of distinct string literals.
func (q *Query) StringCount() int { return int(ts.QueryStringCount(q.q)) }

// StartByteForPattern returns where pattern i begins in the query source.
func (q *Query) StartByteForPattern(i int) int {
	return int(ts.QueryStartByteForPattern(q.q, u32(i)))
}

// EndByteForPattern returns where pattern i ends in the query source.
func (q *Query) EndByteForPattern(i int) int {
	return int(ts.QueryEndByteForPattern(q.q, u32(i)))
}

// IsPatternRooted reports whether pattern i has a single root node.
func (q *Query) IsPatternRooted(i int) bool { return ts.QueryIsPatternRooted(q.q, u32(i)) }

// IsPatternNonLocal reports whether pattern i can match nodes that are
// not direct descendants of a single node.
func (q *Query) IsPatternNonLocal(i int) bool { return ts.QueryIsPatternNonLocal(q.q, u32(i)) }

// IsPatternGuaranteedAtStep reports whether a pattern is certain to match
// once the query has reached the given byte offset of its source.
func (q *Query) IsPatternGuaranteedAtStep(byteOffset int) bool {
	return ts.QueryIsPatternGuaranteedAtStep(q.q, u32(byteOffset))
}

// CaptureName returns the name of capture id.
func (q *Query) CaptureName(id int) string {
	if id >= 0 && id < len(q.names) {
		return q.names[id]
	}
	var n uint32
	return ts.QueryCaptureNameForId(q.q, u32(id), &n)
}

// CaptureNames returns the query's capture names, indexed by capture id.
func (q *Query) CaptureNames() []string {
	return slices.Clone(q.names)
}

// CaptureQuantifier returns how many times capture id may occur in
// pattern.
func (q *Query) CaptureQuantifier(pattern, id int) Quantifier {
	return Quantifier(ts.QueryCaptureQuantifierForId(q.q, u32(pattern), u32(id)))
}

// StringValue returns the string literal with the given id.
func (q *Query) StringValue(id int) string {
	var n uint32
	return ts.QueryStringValueForId(q.q, u32(id), &n)
}

// DisableCapture stops the query from reporting the named capture.
func (q *Query) DisableCapture(name string) {
	ts.QueryDisableCapture(q.q, stringPtr(name), uint32(len(name)))
}

// DisablePattern stops the query from matching pattern i.
func (q *Query) DisablePattern(i int) {
	ts.QueryDisablePattern(q.q, u32(i))
}

// QueryCapture is a node matched by a named capture.
type QueryCapture struct {
	Node  Node
	Index int // capture id; see Query.CaptureName
}

// QueryMatch is one match of one pattern.
type QueryMatch struct {
	ID           int
	PatternIndex int
	Captures     []QueryCapture
}

// QueryCursor runs a [Query] over a subtree. A cursor owns C memory and
// must be closed. It may be reused for many queries.
type QueryCursor struct {
	c ts.QueryCursor

	// t is the tree of the node passed to the current execution, so that
	// matched nodes can report their source text.
	t *Tree

	// ctxID keys the context of an ExecContext run in the callback
	// registry until the next Exec or Close.
	ctxID uintptr
}

// NewQueryCursor returns a cursor with no query executing.
func NewQueryCursor() (*QueryCursor, error) {
	if err := load(); err != nil {
		return nil, err
	}
	c := ts.QueryCursorNew()
	if c == 0 {
		return nil, errors.New("treesitter: ts_query_cursor_new returned NULL")
	}
	return &QueryCursor{c: ts.QueryCursor(c)}, nil
}

// Close frees the cursor.
func (c *QueryCursor) Close() {
	if c.c == 0 {
		return
	}
	c.dropContext()
	ts.QueryCursorDelete(c.c)
	c.c = 0
}

// Exec starts running q over node's subtree. Results come from NextMatch
// or NextCapture.
func (c *QueryCursor) Exec(q *Query, node Node) {
	c.dropContext()
	c.t = node.t
	ts.QueryCursorExec(c.c, q.q, node.n)
}

// ExecContext is like Exec but stops producing results once ctx is done.
// tree-sitter polls the context as it advances, so NextMatch and
// NextCapture return ok false soon after cancellation; callers that need
// to tell cancellation from exhaustion check ctx.Err. The context is
// released by the next Exec, ExecContext, or Close.
func (c *QueryCursor) ExecContext(ctx context.Context, q *Query, node Node) {
	c.dropContext()
	c.t = node.t
	if ctx.Done() == nil {
		ts.QueryCursorExec(c.c, q.q, node.n)
		return
	}
	c.ctxID = register(ctx)
	opts := ts.QueryCursorOptions{Payload: c.ctxID, ProgressCallback: queryProgressCallback()}
	ts.QueryCursorExecWithOptions(c.c, q.q, node.n, &opts)
}

func (c *QueryCursor) dropContext() {
	if c.ctxID != 0 {
		callbacks.Delete(c.ctxID)
		c.ctxID = 0
	}
}

// NextMatch returns the next match, or ok false when there are no more.
func (c *QueryCursor) NextMatch() (m QueryMatch, ok bool) {
	var raw ts.QueryMatch
	if !ts.QueryCursorNextMatch(c.c, &raw) {
		return QueryMatch{}, false
	}
	return fromMatch(raw, c.t), true
}

// NextCapture returns the next capture in document order along with the
// match it belongs to, or ok false when there are no more.
func (c *QueryCursor) NextCapture() (m QueryMatch, captureIndex int, ok bool) {
	var raw ts.QueryMatch
	var idx uint32
	if !ts.QueryCursorNextCapture(c.c, &raw, &idx) {
		return QueryMatch{}, 0, false
	}
	return fromMatch(raw, c.t), int(idx), true
}

// Matches returns an iterator over the remaining matches of the current
// execution, started by Exec or ExecContext. Ranging over it consumes the
// matches; a later NextMatch continues where the loop stopped.
func (c *QueryCursor) Matches() iter.Seq[QueryMatch] {
	return func(yield func(QueryMatch) bool) {
		for {
			m, ok := c.NextMatch()
			if !ok || !yield(m) {
				return
			}
		}
	}
}

// Captures returns an iterator over the remaining captures of the current
// execution in document order, each paired with the match it belongs to.
func (c *QueryCursor) Captures() iter.Seq2[QueryMatch, QueryCapture] {
	return func(yield func(QueryMatch, QueryCapture) bool) {
		for {
			m, i, ok := c.NextCapture()
			if !ok || !yield(m, m.Captures[i]) {
				return
			}
		}
	}
}

// RemoveMatch discards an in-progress match by id, freeing its resources.
func (c *QueryCursor) RemoveMatch(id int) {
	ts.QueryCursorRemoveMatch(c.c, u32(id))
}

// SetByteRange restricts results to nodes within the byte range. It
// reports false if the range is invalid.
func (c *QueryCursor) SetByteRange(start, end int) bool {
	return ts.QueryCursorSetByteRange(c.c, u32(start), u32(end))
}

// SetPointRange restricts results to nodes within the point range. It
// reports false if the range is invalid.
func (c *QueryCursor) SetPointRange(start, end Point) bool {
	return ts.QueryCursorSetPointRange(c.c, toPoint(start), toPoint(end))
}

// SetContainingByteRange restricts results to nodes that contain the byte
// range. It reports false if the range is invalid.
func (c *QueryCursor) SetContainingByteRange(start, end int) bool {
	return ts.QueryCursorSetContainingByteRange(c.c, u32(start), u32(end))
}

// SetContainingPointRange restricts results to nodes that contain the
// point range. It reports false if the range is invalid.
func (c *QueryCursor) SetContainingPointRange(start, end Point) bool {
	return ts.QueryCursorSetContainingPointRange(c.c, toPoint(start), toPoint(end))
}

// SetMatchLimit caps the number of in-progress matches. The default is
// unlimited.
func (c *QueryCursor) SetMatchLimit(limit int) {
	ts.QueryCursorSetMatchLimit(c.c, u32(limit))
}

// MatchLimit returns the cap on in-progress matches.
func (c *QueryCursor) MatchLimit() int {
	return int(ts.QueryCursorMatchLimit(c.c))
}

// DidExceedMatchLimit reports whether matches were dropped because the
// limit was reached.
func (c *QueryCursor) DidExceedMatchLimit() bool {
	return ts.QueryCursorDidExceedMatchLimit(c.c)
}

// SetMaxStartDepth limits how far below the node passed to Exec a pattern
// may start matching. Zero matches only that node.
func (c *QueryCursor) SetMaxStartDepth(depth int) {
	ts.QueryCursorSetMaxStartDepth(c.c, u32(depth))
}

// fromMatch copies a match out of cursor-owned memory, which is valid only
// until the next cursor call.
func fromMatch(raw ts.QueryMatch, t *Tree) QueryMatch {
	m := QueryMatch{ID: int(raw.Id), PatternIndex: int(raw.PatternIndex)}
	if raw.Captures != 0 && raw.CaptureCount > 0 {
		src := unsafe.Slice((*ts.QueryCapture)(cptr(raw.Captures)), raw.CaptureCount)
		m.Captures = make([]QueryCapture, len(src))
		for i, c := range src {
			m.Captures[i] = QueryCapture{Node: Node{t: t, n: c.Node}, Index: int(c.Index)}
		}
	}
	return m
}
