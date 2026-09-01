package treesitter

import (
	"errors"

	"github.com/tmc/lib/treesitter/internal/ts"
)

// LookaheadIterator enumerates the symbols valid in a parse state, which
// is the basis of syntax-aware completion and error messages. It owns C
// memory and must be closed.
type LookaheadIterator struct {
	it ts.LookaheadIterator
}

// NewLookaheadIterator returns an iterator over the symbols valid in
// state. It fails for a state the language does not have.
func NewLookaheadIterator(lang *Language, state StateID) (*LookaheadIterator, error) {
	if lang == nil {
		return nil, errors.New("treesitter: NewLookaheadIterator: nil language")
	}
	it := ts.LookaheadIteratorNew(lang.l, ts.StateId(state))
	if it == 0 {
		return nil, errors.New("treesitter: NewLookaheadIterator: invalid parse state")
	}
	return &LookaheadIterator{it: ts.LookaheadIterator(it)}, nil
}

// Close frees the iterator.
func (it *LookaheadIterator) Close() {
	if it.it == 0 {
		return
	}
	ts.LookaheadIteratorDelete(it.it)
	it.it = 0
}

// ResetState restarts the iterator at another state of the same language.
// It reports false for an invalid state.
func (it *LookaheadIterator) ResetState(state StateID) bool {
	return ts.LookaheadIteratorResetState(it.it, ts.StateId(state))
}

// Reset restarts the iterator at a state of another language. It reports
// false for an invalid state.
func (it *LookaheadIterator) Reset(lang *Language, state StateID) bool {
	return ts.LookaheadIteratorReset(it.it, lang.l, ts.StateId(state))
}

// Language returns the iterator's language.
func (it *LookaheadIterator) Language() *Language {
	return NewLanguage(ts.LookaheadIteratorLanguage(it.it))
}

// Next advances to the next symbol and reports whether there was one.
func (it *LookaheadIterator) Next() bool {
	return ts.LookaheadIteratorNext(it.it)
}

// CurrentSymbol returns the symbol the iterator is on.
func (it *LookaheadIterator) CurrentSymbol() Symbol {
	return Symbol(ts.LookaheadIteratorCurrentSymbol(it.it))
}

// CurrentSymbolName returns the name of the symbol the iterator is on.
func (it *LookaheadIterator) CurrentSymbolName() string {
	return ts.LookaheadIteratorCurrentSymbolName(it.it)
}
