package dsv41

import (
	"fmt"

	"github.com/tmc/lib/examples/ds4-min/dsv41engram"
)

// EngramRows owns request-local sparse rows and transactional hash history.
// It performs no projection. Its immutable layout and table readers are
// borrowed through Close. A request must not call its methods concurrently.
// Prepare stages the next token; Commit publishes it only after the caller's
// entire token evaluation succeeds. Abort discards staging without publishing.
type EngramRows struct {
	layout                 *dsv41engram.Layout
	tables                 [2]*dsv41engram.Table
	history, next          dsv41engram.History
	rows                   [2][]float32
	position, context      int
	prepared, text, closed bool
}

// NewEngramRows binds bounded row readers without expanding the SSD tables.
// The zero value is unusable. Context must be in [1,1048576].
func NewEngramRows(layout *dsv41engram.Layout, tables [2]*dsv41engram.Table, context int) (*EngramRows, error) {
	if layout == nil || tables[0] == nil || tables[1] == nil || context <= 0 || context > 1048576 {
		return nil, fmt.Errorf("invalid engram row source or context")
	}
	s := &EngramRows{layout: layout, tables: tables, context: context}
	for i := range s.rows {
		s.rows[i] = make([]float32, dsv41engram.Columns*dsv41engram.Dim)
	}
	return s, nil
}

// NewEngramRows creates an independent device-consumer row source from this
// validated model. No CPU projection matrix or model-sized copy is created.
func (m *Model) NewEngramRows(context int) (*EngramRows, error) {
	if m == nil || m.catalog == nil {
		return nil, fmt.Errorf("invalid V4.1 model")
	}
	return NewEngramRows(m.layout, m.tables, context)
}

// Position reports the number of committed tokens, never the staged token.
func (s *EngramRows) Position() int {
	if s == nil {
		return 0
	}
	return s.position
}

// Prepare hashes and reads both table sets for exactly the current position.
// Failed reads publish neither history nor position. Masked image tokens still
// advance staged DEAD history, but perform no sparse reads.
func (s *EngramRows) Prepare(token int32, text bool, position int) error {
	if s == nil || s.closed || s.layout == nil || s.prepared || position != s.position || position >= s.context {
		return fmt.Errorf("invalid engram preparation frontier")
	}
	next := s.history
	var ids [2 * dsv41engram.Columns]uint32
	mask := byte(0)
	if text {
		mask = 1
	}
	if err := s.layout.Hash(&next, []int32{token}, []byte{mask}, ids[:]); err != nil {
		return err
	}
	if text {
		for i, table := range s.tables {
			if err := table.ReadRows(ids[i*dsv41engram.Columns:(i+1)*dsv41engram.Columns], s.rows[i]); err != nil {
				return fmt.Errorf("read engram table %d: %w", i, err)
			}
		}
	}
	s.next = next
	s.text = text
	s.prepared = true
	return nil
}

// Rows returns a detached decoded row for layer 1 or 14. A masked token returns
// nil,false,nil. The caller owns the returned 6144 values, already BF16-rounded.
func (s *EngramRows) Rows(layer int) ([]float32, bool, error) {
	if s == nil || s.closed || !s.prepared {
		return nil, false, fmt.Errorf("engram token not prepared")
	}
	index := 0
	switch layer {
	case 1:
	case 14:
		index = 1
	default:
		return nil, false, fmt.Errorf("invalid engram insertion layer")
	}
	if !s.text {
		return nil, false, nil
	}
	return append([]float32(nil), s.rows[index]...), true, nil
}

// Commit publishes exactly one successfully evaluated token. Calling Commit
// twice or before Prepare refuses without advancing history.
func (s *EngramRows) Commit() error {
	if s == nil || s.closed || !s.prepared {
		return fmt.Errorf("engram token not prepared")
	}
	s.history = s.next
	s.position++
	s.prepared = false
	return nil
}

// Abort discards a prepared token without changing the committed frontier.
// It cannot undo device writes; the device owner must invalidate those on error.
func (s *EngramRows) Abort() {
	if s != nil {
		s.prepared = false
		s.text = false
		s.next = dsv41engram.History{}
	}
}

// Reset restarts this request's committed hash history and staged rows.
func (s *EngramRows) Reset() error {
	if s == nil || s.closed || s.layout == nil {
		return fmt.Errorf("closed engram row source")
	}
	s.Abort()
	s.history.Reset()
	s.position = 0
	return nil
}

// Close releases small request storage, not the borrowed table readers.
func (s *EngramRows) Close() error {
	if s != nil {
		s.Abort()
		s.closed = true
		s.rows = [2][]float32{}
		s.layout = nil
		s.tables = [2]*dsv41engram.Table{}
	}
	return nil
}
