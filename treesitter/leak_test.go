package treesitter

import (
	"context"
	"strings"
	"testing"
)

// callbackCount reports how many entries the C callback registry holds.
// Every entry pins a Go value for as long as C may call back into it, so a
// count that grows across parses is a leak.
func callbackCount() int {
	n := 0
	callbacks.Range(func(_, _ any) bool {
		n++
		return true
	})
	return n
}

// TestCallbackRegistryDrains checks that every path that hands a Go value
// to C removes it again, including the ones that end early.
func TestCallbackRegistryDrains(t *testing.T) {
	p, lang := testParser(t)
	src := []byte(matchSrc)

	// Warm up, so one-time registrations are not counted as growth.
	tree, err := p.Parse(src, nil)
	if err != nil {
		t.Fatal(err)
	}
	tree.Close()
	base := callbackCount()

	q, err := NewQuery(lang, `(call_expression function: (identifier) @callee)`)
	if err != nil {
		t.Fatal(err)
	}
	defer q.Close()

	long := longSource()
	for i := 0; i < 20; i++ {
		tree, err := p.Parse(src, nil)
		if err != nil {
			t.Fatal(err)
		}

		if t2, err := p.ParseContext(context.Background(), src, nil); err != nil {
			t.Fatal(err)
		} else {
			t2.Close()
		}

		// A cancelled parse must clean up too.
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if t2, _ := p.ParseContext(ctx, src, nil); t2 != nil {
			t2.Close()
		}

		if t2, err := p.ParseReaderAt(strings.NewReader(long), int64(len(long)), nil); err != nil {
			t.Fatal(err)
		} else {
			t2.Close()
		}

		root := tree.RootNode()
		for range q.Matches(root) {
			break
		}
		for range q.Captures(root) {
		}
		for range q.MatchesContext(context.Background(), root) {
			break
		}

		// A logger registers the parser until it is removed.
		p.SetLogger(func(LogType, string) {})
		if t2, err := p.Parse(src, nil); err != nil {
			t.Fatal(err)
		} else {
			t2.Close()
		}
		p.SetLogger(nil)

		tree.Close()
	}

	if got := callbackCount(); got != base {
		t.Errorf("callback registry holds %d entries, want %d", got, base)
	}
}
