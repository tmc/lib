package treesitter

import (
	"context"
	"iter"
)

// Capture is a node matched by a named capture, with the name resolved.
type Capture struct {
	Name  string
	Node  Node
	Index int // capture id; see [Query.CaptureName]
}

// Match is one match of one query pattern, with its captures named and
// copied out of the cursor. It is the result of [Query.Matches].
type Match struct {
	ID       int
	Pattern  int
	Captures []Capture
}

// Node returns the node of the first capture with the given name, or a
// null node when the match has none.
func (m Match) Node(name string) Node {
	for _, c := range m.Captures {
		if c.Name == name {
			return c.Node
		}
	}
	return Node{}
}

// Nodes returns the nodes of every capture with the given name, in match
// order. A capture inside a repetition can match more than once.
func (m Match) Nodes(name string) []Node {
	var out []Node
	for _, c := range m.Captures {
		if c.Name == name {
			out = append(out, c.Node)
		}
	}
	return out
}

// Text returns the source text of the first capture with the given name,
// or nil when the match has none.
func (m Match) Text(name string) []byte { return m.Node(name).Text() }

// Matches runs the query over root's subtree and yields the matches whose
// text predicates hold. It manages its own cursor, which is released when
// the loop ends, including on an early break.
//
// Use [QueryCursor] directly to reuse a cursor, to restrict the search to
// a byte or point range, or to see matches that predicates reject.
func (q *Query) Matches(root Node) iter.Seq[Match] {
	return func(yield func(Match) bool) {
		c, err := NewQueryCursor()
		if err != nil {
			return
		}
		defer c.Close()
		c.Exec(q, root)
		for {
			raw, ok := c.NextMatch()
			if !ok {
				return
			}
			if !q.Satisfies(raw) {
				continue
			}
			if !yield(q.match(raw)) {
				return
			}
		}
	}
}

// MatchesContext is like [Query.Matches] but stops when ctx is done. It
// yields each match with a nil error, then a single zero match with ctx's
// error if the search was cancelled. When cancellation happens as the last
// match is produced, the error may follow a complete set of results.
func (q *Query) MatchesContext(ctx context.Context, root Node) iter.Seq2[Match, error] {
	return func(yield func(Match, error) bool) {
		c, err := NewQueryCursor()
		if err != nil {
			yield(Match{}, err)
			return
		}
		defer c.Close()
		c.ExecContext(ctx, q, root)
		for {
			raw, ok := c.NextMatch()
			if !ok {
				if err := ctx.Err(); err != nil {
					yield(Match{}, err)
				}
				return
			}
			if !q.Satisfies(raw) {
				continue
			}
			if !yield(q.match(raw), nil) {
				return
			}
		}
	}
}

// Captures runs the query over root's subtree and yields every capture in
// document order, each with the match it belongs to. Highlighting wants
// this order; [Query.Matches] groups by match instead.
func (q *Query) Captures(root Node) iter.Seq2[Match, Capture] {
	return func(yield func(Match, Capture) bool) {
		c, err := NewQueryCursor()
		if err != nil {
			return
		}
		defer c.Close()
		c.Exec(q, root)
		for {
			raw, i, ok := c.NextCapture()
			if !ok {
				return
			}
			if !q.Satisfies(raw) {
				// Drop the whole match so its later captures are not
				// reported either.
				c.RemoveMatch(raw.ID)
				continue
			}
			m := q.match(raw)
			if !yield(m, m.Captures[i]) {
				return
			}
		}
	}
}

// match resolves capture names for a raw match.
func (q *Query) match(raw QueryMatch) Match {
	m := Match{ID: raw.ID, Pattern: raw.PatternIndex}
	if len(raw.Captures) == 0 {
		return m
	}
	m.Captures = make([]Capture, len(raw.Captures))
	for i, c := range raw.Captures {
		m.Captures[i] = Capture{Name: q.CaptureName(c.Index), Node: c.Node, Index: c.Index}
	}
	return m
}
