package cudashadow

import (
	"errors"
	"fmt"
	"sync"
)

// MutationCall records an illegal invocation of an active GPU or driver mutation method.
type MutationCall struct {
	Method string
	Target string
	Args   string
}

// ActiveBackendSpy monitors backend mutation interfaces.
//
// In shadow evaluation mode, all policy actions are strictly observational.
// This spy acts as a verification harness asserting that zero mutation calls
// occur across all normal, cancellation, and error paths.
type ActiveBackendSpy struct {
	mu    sync.Mutex
	calls []MutationCall
}

// NewActiveBackendSpy constructs a fresh backend spy with zero recorded calls.
func NewActiveBackendSpy() *ActiveBackendSpy {
	return &ActiveBackendSpy{}
}

// MutatePTX simulates an active PTX kernel modification call.
func (s *ActiveBackendSpy) MutatePTX(kernelName string, ptx []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, MutationCall{
		Method: "MutatePTX",
		Target: kernelName,
		Args:   fmt.Sprintf("bytes=%d", len(ptx)),
	})
	return nil
}

// SetTimeslice simulates an active scheduler timeslice update.
func (s *ActiveBackendSpy) SetTimeslice(channelID uint32, timesliceUS uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, MutationCall{
		Method: "SetTimeslice",
		Target: fmt.Sprintf("channel=%d", channelID),
		Args:   fmt.Sprintf("timeslice_us=%d", timesliceUS),
	})
	return nil
}

// EvictChunk simulates an active UVM page chunk eviction call.
func (s *ActiveBackendSpy) EvictChunk(chunkAddr uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, MutationCall{
		Method: "EvictChunk",
		Target: fmt.Sprintf("0x%x", chunkAddr),
	})
	return nil
}

// PrefetchPages simulates an active UVM page prefetch request.
func (s *ActiveBackendSpy) PrefetchPages(va uint64, count uint32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, MutationCall{
		Method: "PrefetchPages",
		Target: fmt.Sprintf("0x%x", va),
		Args:   fmt.Sprintf("count=%d", count),
	})
	return nil
}

// Calls returns a copy of all recorded mutation calls.
func (s *ActiveBackendSpy) Calls() []MutationCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]MutationCall, len(s.calls))
	copy(out, s.calls)
	return out
}

// CallCount returns the number of recorded mutation attempts.
func (s *ActiveBackendSpy) CallCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.calls)
}

// AssertZeroMutations verifies that no mutation calls were executed.
// If any mutation call occurred, it returns an error detailing the illegal calls.
func (s *ActiveBackendSpy) AssertZeroMutations() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.calls) == 0 {
		return nil
	}
	return fmt.Errorf("cudashadow: illegal mutation calls detected (%d calls, first: %s on %s)",
		len(s.calls), s.calls[0].Method, s.calls[0].Target)
}

// Reset clears recorded calls.
func (s *ActiveBackendSpy) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = nil
}

// ErrMutationForbidden is returned if any code path mistakenly attempts an active backend mutation.
var ErrMutationForbidden = errors.New("cudashadow: active backend mutation forbidden in shadow mode")
