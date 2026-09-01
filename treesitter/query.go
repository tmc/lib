package treesitter

import (
	"context"
	"errors"
	"fmt"
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
// byte as a zero-based line and column.
type QueryError struct {
	Kind   QueryErrorKind
	Offset uint32
	Row    uint32
	Column uint32
}

func (e *QueryError) Error() string {
	return fmt.Sprintf("treesitter: query %s error at %d:%d (byte %d)", e.Kind, e.Row, e.Column, e.Offset)
}

// newQueryError locates offset within source.
func newQueryError(kind ts.QueryError, source string, offset uint32) *QueryError {
	e := &QueryError{Kind: QueryErrorKind(kind), Offset: offset}
	if int(offset) > len(source) {
		offset = uint32(len(source))
	}
	before := source[:offset]
	e.Row = uint32(strings.Count(before, "\n"))
	e.Column = uint32(len(before) - (strings.LastIndexByte(before, '\n') + 1))
	return e
}

// Query is a compiled set of tree patterns. A query owns C memory and
// must be closed.
type Query struct {
	q ts.Query
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
		return nil, newQueryError(kind, source, offset)
	}
	return &Query{q: ts.Query(q)}, nil
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
func (q *Query) PatternCount() uint32 { return ts.QueryPatternCount(q.q) }

// CaptureCount returns the number of distinct capture names.
func (q *Query) CaptureCount() uint32 { return ts.QueryCaptureCount(q.q) }

// StringCount returns the number of distinct string literals.
func (q *Query) StringCount() uint32 { return ts.QueryStringCount(q.q) }

// StartByteForPattern returns where pattern i begins in the query source.
func (q *Query) StartByteForPattern(i uint32) uint32 { return ts.QueryStartByteForPattern(q.q, i) }

// EndByteForPattern returns where pattern i ends in the query source.
func (q *Query) EndByteForPattern(i uint32) uint32 { return ts.QueryEndByteForPattern(q.q, i) }

// IsPatternRooted reports whether pattern i has a single root node.
func (q *Query) IsPatternRooted(i uint32) bool { return ts.QueryIsPatternRooted(q.q, i) }

// IsPatternNonLocal reports whether pattern i can match nodes that are
// not direct descendants of a single node.
func (q *Query) IsPatternNonLocal(i uint32) bool { return ts.QueryIsPatternNonLocal(q.q, i) }

// IsPatternGuaranteedAtStep reports whether a pattern is certain to match
// once the query has reached the given byte offset of its source.
func (q *Query) IsPatternGuaranteedAtStep(byteOffset uint32) bool {
	return ts.QueryIsPatternGuaranteedAtStep(q.q, byteOffset)
}

// CaptureName returns the name of capture id.
func (q *Query) CaptureName(id uint32) string {
	var n uint32
	return ts.QueryCaptureNameForId(q.q, id, &n)
}

// CaptureQuantifier returns how many times capture id may occur in
// pattern.
func (q *Query) CaptureQuantifier(pattern, id uint32) Quantifier {
	return Quantifier(ts.QueryCaptureQuantifierForId(q.q, pattern, id))
}

// StringValue returns the string literal with the given id.
func (q *Query) StringValue(id uint32) string {
	var n uint32
	return ts.QueryStringValueForId(q.q, id, &n)
}

// DisableCapture stops the query from reporting the named capture.
func (q *Query) DisableCapture(name string) {
	ts.QueryDisableCapture(q.q, stringPtr(name), uint32(len(name)))
}

// DisablePattern stops the query from matching pattern i.
func (q *Query) DisablePattern(i uint32) {
	ts.QueryDisablePattern(q.q, i)
}

// PredicateStepType classifies the steps of a predicate.
type PredicateStepType int32

// Predicate step types. A Done step ends each predicate.
const (
	PredicateStepDone    PredicateStepType = PredicateStepType(ts.QueryPredicateStepDone)
	PredicateStepCapture PredicateStepType = PredicateStepType(ts.QueryPredicateStepCapture)
	PredicateStepString  PredicateStepType = PredicateStepType(ts.QueryPredicateStepString)
)

// PredicateStep is one token of a predicate such as (#eq? @a "b"). ValueID
// is a capture id for a Capture step and a string id for a String step.
type PredicateStep struct {
	Type    PredicateStepType
	ValueID uint32
}

// PredicatesForPattern returns the predicate steps of pattern i.
// tree-sitter does not evaluate predicates; callers interpret them using
// [Query.CaptureName] and [Query.StringValue].
func (q *Query) PredicatesForPattern(i uint32) []PredicateStep {
	var n uint32
	p := ts.QueryPredicatesForPattern(q.q, i, &n)
	if p == 0 || n == 0 {
		return nil
	}
	src := unsafe.Slice((*ts.QueryPredicateStep)(cptr(p)), n)
	out := make([]PredicateStep, n)
	for j, s := range src {
		out[j] = PredicateStep{Type: PredicateStepType(s.Type), ValueID: s.ValueId}
	}
	return out
}

// QueryCapture is a node matched by a named capture.
type QueryCapture struct {
	Node  Node
	Index uint32 // capture id; see Query.CaptureName
}

// QueryMatch is one match of one pattern.
type QueryMatch struct {
	ID           uint32
	PatternIndex uint16
	Captures     []QueryCapture
}

// QueryCursor runs a [Query] over a subtree. A cursor owns C memory and
// must be closed. It may be reused for many queries.
type QueryCursor struct {
	c ts.QueryCursor

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
	ts.QueryCursorExec(c.c, q.q, node.n)
}

// ExecContext is like Exec but stops producing results once ctx is done.
// tree-sitter polls the context as it advances, so NextMatch and
// NextCapture return ok false soon after cancellation; callers that need
// to tell cancellation from exhaustion check ctx.Err. The context is
// released by the next Exec, ExecContext, or Close.
func (c *QueryCursor) ExecContext(ctx context.Context, q *Query, node Node) {
	c.dropContext()
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
	return fromMatch(raw), true
}

// NextCapture returns the next capture in document order along with the
// match it belongs to, or ok false when there are no more.
func (c *QueryCursor) NextCapture() (m QueryMatch, captureIndex uint32, ok bool) {
	var raw ts.QueryMatch
	var idx uint32
	if !ts.QueryCursorNextCapture(c.c, &raw, &idx) {
		return QueryMatch{}, 0, false
	}
	return fromMatch(raw), idx, true
}

// RemoveMatch discards an in-progress match by id, freeing its resources.
func (c *QueryCursor) RemoveMatch(id uint32) {
	ts.QueryCursorRemoveMatch(c.c, id)
}

// SetByteRange restricts results to nodes within the byte range. It
// reports false if the range is invalid.
func (c *QueryCursor) SetByteRange(start, end uint32) bool {
	return ts.QueryCursorSetByteRange(c.c, start, end)
}

// SetPointRange restricts results to nodes within the point range. It
// reports false if the range is invalid.
func (c *QueryCursor) SetPointRange(start, end Point) bool {
	return ts.QueryCursorSetPointRange(c.c, toPoint(start), toPoint(end))
}

// SetContainingByteRange restricts results to nodes that contain the byte
// range. It reports false if the range is invalid.
func (c *QueryCursor) SetContainingByteRange(start, end uint32) bool {
	return ts.QueryCursorSetContainingByteRange(c.c, start, end)
}

// SetContainingPointRange restricts results to nodes that contain the
// point range. It reports false if the range is invalid.
func (c *QueryCursor) SetContainingPointRange(start, end Point) bool {
	return ts.QueryCursorSetContainingPointRange(c.c, toPoint(start), toPoint(end))
}

// SetMatchLimit caps the number of in-progress matches. The default is
// unlimited.
func (c *QueryCursor) SetMatchLimit(limit uint32) {
	ts.QueryCursorSetMatchLimit(c.c, limit)
}

// MatchLimit returns the cap on in-progress matches.
func (c *QueryCursor) MatchLimit() uint32 {
	return ts.QueryCursorMatchLimit(c.c)
}

// DidExceedMatchLimit reports whether matches were dropped because the
// limit was reached.
func (c *QueryCursor) DidExceedMatchLimit() bool {
	return ts.QueryCursorDidExceedMatchLimit(c.c)
}

// SetMaxStartDepth limits how far below the node passed to Exec a pattern
// may start matching. Zero matches only that node.
func (c *QueryCursor) SetMaxStartDepth(depth uint32) {
	ts.QueryCursorSetMaxStartDepth(c.c, depth)
}

// fromMatch copies a match out of cursor-owned memory, which is valid only
// until the next cursor call.
func fromMatch(raw ts.QueryMatch) QueryMatch {
	m := QueryMatch{ID: raw.Id, PatternIndex: raw.PatternIndex}
	if raw.Captures != 0 && raw.CaptureCount > 0 {
		src := unsafe.Slice((*ts.QueryCapture)(cptr(raw.Captures)), raw.CaptureCount)
		m.Captures = make([]QueryCapture, len(src))
		for i, c := range src {
			m.Captures[i] = QueryCapture{Node: Node{n: c.Node}, Index: c.Index}
		}
	}
	return m
}
